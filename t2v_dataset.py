import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import pandas as pd
import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torchvision import transforms

from utils.dataset_utils import preprocess_video_with_resize

LOG_NAME = "t2v_trainer"
LOG_LEVEL = "INFO"
logger = get_logger(LOG_NAME, LOG_LEVEL)


class T2VContrastDatasetWithResize:
    """
    参考 PromptSafe 的 PromptPairDataset：从 CSV 读取列 prompt/rewritten_prompt，
    根据 placeholder_token 与 position 生成 pseudo_prompt/pseudo_rewritten，
    同时读取视频并编码为 latent。

    期望 CSV 列：prompt, rewritten_prompt
    期望视频列表文件：与 Args.video_column 指向的文件相同格式（逐行相对路径）。
    行数需与 CSV 行数一致，一一对应。
    """

    def __init__(
        self,
        data_root: str,
        device: torch.device,
        trainer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.data_root = data_root
        self.df = pd.read_csv(data_root)
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.trainer = trainer


        self.train_resolution = getattr(trainer.args, "train_resolution", "17x240x360")
        self.max_num_frames = int(self.train_resolution.split("x")[0])
        self.height = int(self.train_resolution.split("x")[1])
        self.width = int(self.train_resolution.split("x")[2])

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

        # 占位符与位置从 args 读取（若无则使用默认与 PromptSafe 一致）
        self.placeholder_token = getattr(trainer.args, "placeholder_token", "<safety>")
        self.position = getattr(trainer.args, "position", "start")  # "start" or "end"

        cache_dir = Path(os.path.dirname(data_root)) / "cache"
        self.prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        self.video_latent_dir = (
            cache_dir
            / "video_latent"
            / self.trainer.args.trainer
            / self.train_resolution
        )
        self.prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.video_latent_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        prompt = str(row["prompt"])
        rewritten_prompt = str(row["rewritten_prompt"])
        video_path = str(row["video_path"])

        if self.position == "start":
            pseudo_prompt = f"{self.placeholder_token} {prompt}"
            pseudo_rewritten = f"{self.placeholder_token} {rewritten_prompt}"
        else:
            pseudo_prompt = f"{prompt} {self.placeholder_token}"
            pseudo_rewritten = f"{rewritten_prompt} {self.placeholder_token}"

        # 文本嵌入缓存
        def get_text_embedding(text: str) -> torch.Tensor:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            path = self.prompt_embeddings_dir / f"{text_hash}.safetensors"
            if path.exists():
                return load_file(path)["prompt_embedding"]
            emb = self.encode_text(text)[0].to("cpu")  # [1, L, D] -> [L, D]
            save_file({"prompt_embedding": emb}, path)
            logger.debug(
                f"Saved prompt embedding to {path}", main_process_only=False
            )
            return emb

        emb_toxic = get_text_embedding(prompt)
        emb_rewritten = get_text_embedding(rewritten_prompt)
        emb_pseudo = get_text_embedding(pseudo_prompt)
        emb_benign = get_text_embedding(pseudo_rewritten)

        # 视频 latent 缓存
        encoded_video_path = self.video_latent_dir / (Path(video_path).stem + ".safetensors")
        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
        else:
            video_path_obj = Path(os.path.join(os.path.dirname(self.data_root), video_path))
            frames = preprocess_video_with_resize(
                video_path_obj, self.max_num_frames, self.height, self.width
            )
            frames = torch.stack([self.__frame_transform(f) for f in frames], dim=0)
            frames = frames.to(self.device)
            frames = frames.unsqueeze(0)  # [1, F, C, H, W]
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()  # [1, C, F, H, W]
            encoded_video = self.encode_video(frames)[0].to("cpu")  # [C, F, H, W]
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.debug(
                f"Saved encoded video to {encoded_video_path}", main_process_only=False
            )

        return {
            "encoded_video": encoded_video,
            # embeddings (for LoRA 对齐/TwoLoss 训练路径)
            "prompt_embedding_toxic": emb_toxic,
            "prompt_embedding_rewritten": emb_rewritten,
            "prompt_embedding_pseudo": emb_pseudo,
            "prompt_embedding_benign": emb_benign,
            # raw prompts (for 软词向量训练路径)
            "prompt": prompt,
            "rewritten_prompt": rewritten_prompt,
            "pseudo_prompt": pseudo_prompt,
            "pseudo_rewritten": pseudo_rewritten,
        }


