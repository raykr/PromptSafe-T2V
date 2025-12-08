import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import os


class SafeTextEncoderTrainer:
    def __init__(self, model_name, placeholder_token="<safe>", initializer_token="safe", num_vectors=1, device="cuda", log_dir="runs"):
        self.device = device
        self.num_vectors = num_vectors
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token
        self.log_dir = log_dir
        
        # 创建TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 1. 加载预训练 T5Encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder")
        self.text_encoder.to(device)
        
        # 2. 设置 placeholder tokens
        self.setup_placeholder_tokens()

        # 3. 冻结 text_model 的参数
        self.text_encoder.requires_grad_(False)
        

        # 只训练 soft token embedding
        # 创建可训练的soft token embedding
        self.soft_token_embedding = self.text_encoder.get_input_embeddings().weight[self.placeholder_token_ids].clone().detach().requires_grad_(True)
        self.optimizer = torch.optim.AdamW([self.soft_token_embedding], lr=5e-4)

        

    def encode(self, texts):
        token_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").input_ids
        token_ids = token_ids.to(self.device)  # 将token_ids移动到正确的设备
        return self.text_encoder(token_ids)[0]  # [B, L, D]

    def training_step(self, malicious, rewritten, benign=None, lambda_benign=0.1):
        # 1. 加上 <safe> soft token
        malicious_safe = [f"{self.placeholder_token} {t}" for t in malicious]

        # 2. 编码 - 使用自定义embedding
        emb_mal_safe = self._encode_with_soft_token(malicious_safe).mean(dim=1)  # [B, D]
        emb_rewritten = self.encode(rewritten).mean(dim=1)  # [B, D]

        # 3. 主 loss：对齐 malicious+<safe> 到 rewritten
        align_loss = F.mse_loss(emb_mal_safe, emb_rewritten)

        # 4. benign consistency（可选）
        if benign is not None:
            emb_benign = self.encode(benign).mean(dim=1)
            emb_benign_safe = self._encode_with_soft_token([f"{self.placeholder_token} {t}" for t in benign]).mean(dim=1)
            benign_loss = F.mse_loss(emb_benign, emb_benign_safe)
        else:
            benign_loss = torch.tensor(0.0, device=self.device)

        loss = align_loss + lambda_benign * benign_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "align_loss": align_loss.item(), "benign_loss": benign_loss.item()}
    
    def train(self, malicious, rewritten, benign=None, lambda_benign=0.1, num_steps=1000, log_interval=10, save_interval=100):
        """训练方法，包含进度条和TensorBoard日志记录"""
        print(f"🚀 开始训练，共 {num_steps} 步...")
        print(f"📊 训练参数: lambda_benign={lambda_benign}, log_interval={log_interval}")
        print(f"📈 TensorBoard日志保存到: {self.log_dir}")
        print("-" * 60)
        
        # 初始化统计信息
        total_loss = 0.0
        total_align_loss = 0.0
        total_benign_loss = 0.0
        best_loss = float('inf')
        
        # 记录训练开始时间
        self.writer.add_text("训练配置", f"num_steps={num_steps}, lambda_benign={lambda_benign}, log_interval={log_interval}")
        
        # 创建进度条
        pbar = tqdm(range(num_steps), desc="训练进度", unit="step", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        start_time = time.time()
        
        for step in pbar:
            # 执行训练步骤
            logs = self.training_step(malicious, rewritten, benign, lambda_benign)
            
            # 更新统计信息
            total_loss += logs["loss"]
            total_align_loss += logs["align_loss"]
            total_benign_loss += logs["benign_loss"]
            
            # 更新最佳loss
            if logs["loss"] < best_loss:
                best_loss = logs["loss"]
            
            # 记录到TensorBoard
            self.writer.add_scalar("Loss/Total", logs["loss"], step)
            self.writer.add_scalar("Loss/Align", logs["align_loss"], step)
            self.writer.add_scalar("Loss/Benign", logs["benign_loss"], step)
            self.writer.add_scalar("Loss/Best", best_loss, step)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Learning_Rate", current_lr, step)
            
            # 记录平均损失
            avg_loss = total_loss / (step + 1)
            avg_align = total_align_loss / (step + 1)
            avg_benign = total_benign_loss / (step + 1)
            self.writer.add_scalar("Loss/Average_Total", avg_loss, step)
            self.writer.add_scalar("Loss/Average_Align", avg_align, step)
            self.writer.add_scalar("Loss/Average_Benign", avg_benign, step)
            
            # 更新进度条描述
            pbar.set_postfix({
                'Loss': f"{logs['loss']:.6f}",
                'Align': f"{logs['align_loss']:.6f}",
                'Benign': f"{logs['benign_loss']:.6f}",
                'Best': f"{best_loss:.6f}"
            })
            
            # 定期打印详细日志
            if step % log_interval == 0:
                elapsed_time = time.time() - start_time
                
                print(f"\n📈 步骤 {step:4d} | "
                      f"Loss: {logs['loss']:.6f} (avg: {avg_loss:.6f}) | "
                      f"Align: {logs['align_loss']:.6f} (avg: {avg_align:.6f}) | "
                      f"Benign: {logs['benign_loss']:.6f} (avg: {avg_benign:.6f}) | "
                      f"时间: {elapsed_time:.1f}s")
                
                # 记录训练速度
                steps_per_sec = (step + 1) / elapsed_time
                self.writer.add_scalar("Training/Speed", steps_per_sec, step)
            
            # 定期保存模型
            if step % save_interval == 0 and step > 0:
                checkpoint_path = f"checkpoint_step_{step}.safetensors"
                self.save_soft_embedding(checkpoint_path)
                print(f"💾 已保存检查点: {checkpoint_path}")
                
                # 记录检查点保存
                self.writer.add_text("检查点", f"步骤 {step}: {checkpoint_path}")
        
        # 训练完成统计
        total_time = time.time() - start_time
        avg_loss = total_loss / num_steps
        avg_align = total_align_loss / num_steps
        avg_benign = total_benign_loss / num_steps
        
        # 记录最终统计到TensorBoard
        self.writer.add_text("训练结果", f"总训练时间: {total_time:.2f}秒")
        self.writer.add_scalar("Final/Average_Total_Loss", avg_loss, num_steps)
        self.writer.add_scalar("Final/Average_Align_Loss", avg_align, num_steps)
        self.writer.add_scalar("Final/Average_Benign_Loss", avg_benign, num_steps)
        self.writer.add_scalar("Final/Best_Loss", best_loss, num_steps)
        self.writer.add_scalar("Final/Total_Time", total_time, num_steps)
        
        print("\n" + "="*60)
        print("🎉 训练完成！")
        print(f"⏱️  总训练时间: {total_time:.2f}秒")
        print(f"📊 最终统计:")
        print(f"   - 平均总Loss: {avg_loss:.6f}")
        print(f"   - 平均对齐Loss: {avg_align:.6f}")
        print(f"   - 平均良性Loss: {avg_benign:.6f}")
        print(f"   - 最佳Loss: {best_loss:.6f}")
        print(f"📈 TensorBoard日志已保存到: {self.log_dir}")
        print("="*60)
    
    def _encode_with_soft_token(self, texts):
        """使用soft token进行编码"""
        token_ids = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # 获取原始embedding矩阵
        embedding_matrix = self.text_encoder.get_input_embeddings().weight
        
        # 创建包含soft token的embedding矩阵
        custom_embeddings = embedding_matrix.clone()
        custom_embeddings[self.placeholder_token_ids] = self.soft_token_embedding
        
        # 手动计算embeddings
        inputs_embeds = F.embedding(token_ids, custom_embeddings)
        
        # 通过T5 encoder
        encoder_outputs = self.text_encoder.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=None,  # 简化处理
        )
        
        return encoder_outputs.last_hidden_state

    def save_soft_embedding(self, path="learned_embeds.safetensors"):
        from safetensors.torch import save_file

        state_dict = {
            self.placeholder_token: self.soft_token_embedding.detach().cpu()
        }
        save_file(state_dict, path)
        print(f"✅ Soft token embedding saved to {path}")
    
    def close_tensorboard(self):
        """关闭TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print(f"📈 TensorBoard writer已关闭")

    def setup_placeholder_tokens(self):
        placeholder_tokens = [self.placeholder_token]
        for i in range(1, self.num_vectors):
            placeholder_tokens.append(f"{self.placeholder_token}_{i}")

        num_added_tokens = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != self.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {self.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        token_ids = self.tokenizer.encode(self.initializer_token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        self.initializer_token_id = token_ids[0]
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in self.placeholder_token_ids:
                token_embeds[token_id] = token_embeds[self.initializer_token_id].clone()


trainer = SafeTextEncoderTrainer(
    model_name="/home/beihang/jzl/models/zai-org/CogVideoX-2b",
    placeholder_token="<safe>",
    initializer_token="safe",
    num_vectors=16,
    device="cuda",
    log_dir="runs/safe_text_encoder_training"
)

# 模拟数据
malicious = ["a nude woman is running in the street"]
rewritten = ["a woman with red clothes is running in the street"]
benign = ["a man is running in the park"]

# 使用新的训练方法（包含进度条）
trainer.train(
    malicious=malicious,
    rewritten=rewritten,
    benign=None,
    lambda_benign=0.1,
    num_steps=10000,
    log_interval=50,  # 每50步打印一次详细日志
    save_interval=500  # 每200步保存一次检查点
)

# 保存最终模型
trainer.save_soft_embedding("learned_safe.safetensors")

# 关闭TensorBoard writer
trainer.close_tensorboard()
