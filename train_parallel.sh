CUDA_VISIBLE_DEVICES=4 python train_adapter.py --category sexual &
CUDA_VISIBLE_DEVICES=4 python train_adapter.py --category violent &
CUDA_VISIBLE_DEVICES=4 python train_adapter.py --category political &
CUDA_VISIBLE_DEVICES=4 python train_adapter.py --category disturbing &
wait