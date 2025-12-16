CUDA_VISIBLE_DEVICES=0 python train_adapter.py --category sexual &
CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category violent &
wait
CUDA_VISIBLE_DEVICES=0 python train_adapter.py --category political &
CUDA_VISIBLE_DEVICES=1 python train_adapter.py --category disturbing &
wait