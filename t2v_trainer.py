from abc import abstractmethod
import json
import logging
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple
import diffusers
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import upload_folder, create_repo
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm.auto import tqdm
from transformers import T5EncoderModel, AutoTokenizer
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.optimization import get_scheduler
import transformers
from t2v_dataset import T2VContrastDatasetWithResize
from utils.log_utils import save_progress, log_validation, save_model_card
from diffusers.utils import check_min_version, is_wandb_available
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from utils.model_utils import register

logger = get_logger(__name__, log_level="INFO")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_accelerator()
        self.setup_logger()
        self.setup_models()
        self.setup_memory()
        self.setup_placeholder_tokens()
        self.setup_optimizer()
        self.setup_dataset()
        self.setup_scheduler()
        self.setup_training_state()

    def setup_accelerator(self):
        if self.args.seed is not None:
            set_seed(self.args.seed)

        logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_logger(self):
        if self.args.report_to == "wandb":
            if self.args.hub_token is not None:
                raise ValueError(
                    "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                )
            
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

            if self.args.push_to_hub:    
                self.repo_id = create_repo(
                    repo_id=self.args.hub_model_id or Path(self.args.output_dir).name, exist_ok=True, token=self.args.hub_token
                ).repo_id

        # save all the arguments in a config file in the output directory
        args_dict = vars(self.args)
        with open(os.path.join(self.args.output_dir, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    def setup_models(self):
        # Load tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.args.revision)
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="transformer")
        self.noise_scheduler = CogVideoXDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")

        if self.args.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.transformer.train()
            self.text_encoder.gradient_checkpointing_enable()
            self.transformer.enable_gradient_checkpointing()

    def setup_memory(self):
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            
        # Enable memory efficient attention
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.transformer.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # Additional memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def setup_placeholder_tokens(self):
        placeholder_tokens = [self.args.placeholder_token]
        for i in range(1, self.args.num_vectors):
            placeholder_tokens.append(f"{self.args.placeholder_token}_{i}")

        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != self.args.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {self.args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        token_ids = self.tokenizer.encode(self.args.initializer_token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        self.initializer_token_id = token_ids[0]
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in self.placeholder_token_ids:
                token_embeds[token_id] = token_embeds[self.initializer_token_id].clone()

        # Freeze models
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.text_encoder.encoder.requires_grad_(False)
        self.text_encoder.encoder.final_layer_norm.requires_grad_(False)
        self.text_encoder.shared.requires_grad_(False)

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [{
                "params": self.text_encoder.get_input_embeddings().parameters(),
                "lr": self.args.learning_rate,
            }],
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

    def setup_dataset(self):
        self.train_dataset = T2VContrastDatasetWithResize(
            data_root=str(self.args.train_data_csv),
            device=self.accelerator.device,
            trainer=self,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def setup_scheduler(self):
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch * self.accelerator.num_processes)

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
        )

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes
            )

    def setup_training_state(self):
        self.text_encoder.train()
        self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # Move transformer and vae to device with proper memory management
        self.transformer = self.accelerator.prepare(self.transformer)
        self.vae = self.accelerator.prepare(self.vae)

        self.orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("textual_inversion", config=vars(self.args))

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos": [],
            "prompt": [],
            "rewritten_prompt": [],
            "pseudo_prompt": [],
            "pseudo_rewritten": [],
        }

        for sample in samples:
            ret["encoded_videos"].append(sample["encoded_video"])  # [C, F, H, W]
            ret["prompt"].append(sample["prompt"])  # [L, D]
            ret["rewritten_prompt"].append(sample["rewritten_prompt"])  # [L, D]
            ret["pseudo_prompt"].append(sample["pseudo_prompt"])  # [L, D]
            ret["pseudo_rewritten"].append(sample["pseudo_rewritten"])  # [L, D]

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])  # [B, C, F, H, W]
        # 将 [L, D] -> [B, L, D]
        # for key in [
        #     "prompt",
        #     "rewritten_prompt",
        #     "pseudo_prompt",
        #     "pseudo_rewritten",
        # ]:
        #     ret[key] = torch.stack(ret[key])

        return ret

    def train(self):
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Device count = {self.accelerator.num_processes}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel={self.args.train_batch_size}, distributed={self.accelerator.num_processes}, accumulation={self.args.gradient_accumulation_steps}) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        self.global_step = 0
        self.first_epoch = 0
        self.loss_list = []

        if self.args.resume_from_checkpoint:
            self.load_checkpoint()

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.first_epoch, self.args.num_train_epochs):
            self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.text_encoder):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.update_embeddings()

                if self.accelerator.sync_gradients:
                    self.handle_sync_gradients(progress_bar)
                
                # 记录进度信息
                logs = {
                    "loss": loss.detach().cpu().item(),
                    "step": self.global_step,
                    "epoch": self.global_step // len(self.train_dataloader),
                    "progress": self.global_step / self.args.max_train_steps
                }
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.args.max_train_steps:
                    break

        self.save_final_model()

    def update_embeddings(self):
        index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
        index_no_updates[min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1] = False

        with torch.no_grad():
            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                index_no_updates
            ] = self.orig_embeds_params[index_no_updates]

    def handle_sync_gradients(self, progress_bar):
        progress_bar.update(1)
        self.global_step += 1

        if self.global_step % self.args.save_steps == 0:
            self.save_checkpoint()

        if self.accelerator.is_main_process:
            if self.global_step % self.args.checkpointing_steps == 0:
                self.save_state()
            
    def load_checkpoint(self):
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            self.global_step = 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.args.output_dir, path))
            self.global_step = int(path.split("-")[1])
            self.first_epoch = self.global_step // math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)

    def save_checkpoint(self):
        weight_name = (
            f"learned_embeds-steps-{self.global_step}.bin"
            if self.args.no_safe_serialization
            else f"learned_embeds-steps-{self.global_step}.safetensors"
        )
        save_path = os.path.join(self.args.output_dir, weight_name)
        save_progress(
            self.text_encoder,
            self.placeholder_token_ids,
            self.accelerator,
            self.args,
            save_path,
            safe_serialization=not self.args.no_safe_serialization,
        )
        
    def save_state(self):
        if self.args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    def run_validation(self):
        images = log_validation(
            self.text_encoder,
            self.tokenizer,
            self.transformer,
            self.vae,
            self.args,
            self.accelerator,
            self.weight_dtype,
            self.global_step // math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        )
        # Save validation set images
        os.makedirs(os.path.join(self.args.output_dir, 'validation'), exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(self.args.output_dir, 'validation', f"{self.global_step}_{i}.png"))

    def save_final_model(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.args.push_to_hub and not self.args.save_as_full_pipeline:
                logger.warning("Enabling full model saving because --push_to_hub=True was specified.")
                save_full_model = True
            else:
                save_full_model = self.args.save_as_full_pipeline

            if save_full_model:
                pipeline = CogVideoXPipeline(
                    tokenizer=self.tokenizer,
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=self.noise_scheduler,
                )
                pipeline.save_pretrained(self.args.output_dir)

            weight_name = "learned_embeds.bin" if self.args.no_safe_serialization else "learned_embeds.safetensors"
            save_path = os.path.join(self.args.output_dir, weight_name)
            save_progress(
                self.text_encoder,
                self.placeholder_token_ids,
                self.accelerator,
                self.args,
                save_path,
                safe_serialization=not self.args.no_safe_serialization,
            )

            if self.args.push_to_hub:
                save_model_card(
                    self.repo_id,
                    images=[],
                    base_model=self.args.pretrained_model_name_or_path,
                    repo_folder=self.args.output_dir,
                )
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()

        # Save the loss information
        loss_df = pd.DataFrame(self.loss_list)
        loss_df.to_csv(os.path.join(self.args.output_dir, "loss.csv")) 

    @abstractmethod
    def training_step(self, batch):
        pass

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin
    
    
class ThreeLossTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        # 提取 prompt pair
        toxic_prompt = batch["prompt"]
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        pseudo_benign = batch["pseudo_rewritten"]

        z_t, timesteps = self.init_latent(len(toxic_prompt))

        # 预测噪声
        noise_pred_toxic, _ = self.predict_noise(toxic_prompt, z_t, timesteps)
        noise_pred_rewritten, _ = self.predict_noise(safe_prompt, z_t, timesteps)
        noise_pred_pseudo, _ = self.predict_noise(pseudo_toxic, z_t, timesteps)
        noise_pred_benign, _ = self.predict_noise(pseudo_benign, z_t, timesteps)

        dist_ps_rw = F.mse_loss(noise_pred_pseudo, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        dist_ps_pt = F.mse_loss(noise_pred_pseudo, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_rw_or = F.mse_loss(noise_pred_rewritten, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_bn_rw = F.mse_loss(noise_pred_benign, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        margin = self.args.margin_coef * dist_rw_or.detach()  # detach 是为了不反传梯度

        # 计算 triplet loss
        triplet_loss = F.relu(dist_ps_rw - dist_ps_pt + margin).mean() * 10 # 拉平数量级
        align_loss = dist_ps_rw
        benign_loss = dist_bn_rw

        loss = self.args.lambda_align * align_loss + self.args.lambda_triplet * triplet_loss + self.args.lambda_benign * benign_loss

        # 记录日志
        self.accelerator.log({
            "loss/total": loss.detach().cpu().item(),
            "loss/align": align_loss.detach().cpu().item(),
            "loss/triplet": triplet_loss.detach().cpu().item(),
            "loss/benign": benign_loss.detach().cpu().item(),
            "dist/pseudo_rewritten": dist_ps_rw.detach().cpu().item(),
            "dist/pseudo_toxic": dist_ps_pt.detach().cpu().item(),
            "dist/rewritten_origin": dist_rw_or.detach().cpu().item(),
            "dist/benign_rewritten": dist_bn_rw.detach().cpu().item(),
            "train/margin": margin,
            "train/learning_rate": self.args.learning_rate,
        }, step=self.global_step)

        # Record loss info
        self.loss_list.append({
            "loss": loss.detach().cpu().item(),
            "step": self.global_step
        })

        return loss

class TwoLossTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        # 提取 prompt pair
        toxic_prompt = batch["prompt"]
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        pseudo_benign = batch["pseudo_rewritten"]
        latent = batch["encoded_videos"]


        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.transformer.config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        toxic_prompt_embedding = self.encode_text(toxic_prompt)
        safe_prompt_embedding = self.encode_text(safe_prompt)
        pseudo_prompt_embedding = self.encode_text(pseudo_toxic)
        pseudo_benign_embedding = self.encode_text(pseudo_benign)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.noise_scheduler.add_noise(latent, noise, timesteps)


        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        transformer_config = self.transformer.config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        noise_pred_toxic = self.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=toxic_prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        noise_pred_rewritten = self.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=safe_prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        noise_pred_pseudo = self.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=pseudo_prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]    

        noise_pred_benign = self.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=pseudo_benign_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Compute losses
        dist_ps_rw = F.mse_loss(noise_pred_pseudo, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        dist_ps_pt = F.mse_loss(noise_pred_pseudo, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_rw_or = F.mse_loss(noise_pred_rewritten, noise_pred_toxic, reduction='none').mean(dim=(1,2,3)).mean()
        dist_bn_rw = F.mse_loss(noise_pred_benign, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()
        margin = self.args.margin_coef * dist_rw_or.detach()  # detach 是为了不反传梯度

        # 计算 triplet loss
        triplet_loss = F.relu(dist_ps_rw - dist_ps_pt + margin).mean() * 10 # 拉平数量级
        benign_loss = dist_bn_rw

        loss = self.args.lambda_triplet * triplet_loss + (1 - self.args.lambda_triplet) * benign_loss

        # 记录日志
        self.accelerator.log({
            "loss/total": loss.detach().cpu().item(),
            "loss/triplet": triplet_loss.detach().cpu().item(),
            "loss/benign": benign_loss.detach().cpu().item(),
            "dist/pseudo_rewritten": dist_ps_rw.detach().cpu().item(),
            "dist/pseudo_toxic": dist_ps_pt.detach().cpu().item(),
            "dist/rewritten_origin": dist_rw_or.detach().cpu().item(),
            "dist/benign_rewritten": dist_bn_rw.detach().cpu().item(),
            "train/margin": margin,
            "train/learning_rate": self.args.learning_rate,
        }, step=self.global_step)

        # Record loss info
        self.loss_list.append({
            "loss": loss.detach().cpu().item(),
            "step": self.global_step
        })

        return loss

class OneLossTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def training_step(self, batch):
        # 提取 prompt pair
        safe_prompt = batch["rewritten_prompt"]
        pseudo_toxic = batch["pseudo_prompt"]
        latent = batch["encoded_videos"]


        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.transformer.config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        safe_prompt_embedding = self.encode_text(safe_prompt)
        pseudo_prompt_embedding = self.encode_text(pseudo_toxic)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # Add noise to latent
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.noise_scheduler.add_noise(latent, noise, timesteps)


        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        transformer_config = self.transformer.config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise
        noise_pred_rewritten = self.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=safe_prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        noise_pred_pseudo = self.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=pseudo_prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]    

        # Compute losses
        dist_ps_rw = F.mse_loss(noise_pred_pseudo, noise_pred_rewritten, reduction='none').mean(dim=(1,2,3)).mean()

        loss = dist_ps_rw

        # 记录日志
        self.accelerator.log({
            "loss/total": loss.detach().cpu().item(),
            "dist/pseudo_rewritten": dist_ps_rw.detach().cpu().item(),
            "train/learning_rate": self.args.learning_rate,
        }, step=self.global_step)

        # Record loss info
        self.loss_list.append({
            "loss": loss.detach().cpu().item(),
            "step": self.global_step
        })

        return loss
    