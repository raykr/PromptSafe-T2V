# 并行训练CogVideoX-2b T5Encoder模型
# CUDA_VISIBLE_DEVICES=0 python train_adapter.py --category sexual &
# CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category violent &
# wait
# CUDA_VISIBLE_DEVICES=0 python train_adapter.py --category political &
# CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category disturbing &
# wait


# 并行训练Wan2.1-T2V-1.3B-Diffusers T5Encoder模型
# CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category sexual --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --adapter_path "checkpoints/wan2.1-t2v-1.3b-diffusers/sexual/safe_adapter.pt" &
# wait
# CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category violent --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --adapter_path "checkpoints/wan2.1-t2v-1.3b-diffusers/violent/safe_adapter.pt" &
# wait
CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category political --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --adapter_path "checkpoints/wan2.1-t2v-1.3b-diffusers/political/safe_adapter.pt" --max_length 256 &
wait
# CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category disturbing --model_path "/home/raykr/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --adapter_path "checkpoints/wan2.1-t2v-1.3b-diffusers/disturbing/safe_adapter.pt" &
# wait