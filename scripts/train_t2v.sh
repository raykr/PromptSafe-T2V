# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Disable PyTorch Dynamo to avoid conflicts with gradient checkpointing
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1

train() {
  local TYPE=$1
  local STEP=$2
  local DATETIME=$3
  local TRAINER=$4
  local EXTRA=$5
  local MODEL_PATH=$6

  OUR_DIR="./out/$DATETIME-$TRAINER$EXTRA"
  CUDA_DEVICE=0

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -m accelerate.commands.accelerate_cli launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --main_process_port=29501 \
    ./main.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --train_data_csv="./datasets/train/caption.csv" \
    --placeholder_token="<${TYPE}>" \
    --initializer_token="safe" \
    --position="start" \
    --resolution=256 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=$STEP \
    --learning_rate=5.0e-04 \
    --lambda_lr=1e-3 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="$OUR_DIR" \
    --num_vectors=1 \
    --seed=42 \
    --lambda_align=0.5 \
    --lambda_triplet=0.9 \
    --lambda_benign=0.1 \
    --margin_coef=0.1 \
    --trainer="$TRAINER" \
    --resume_from_checkpoint "latest" \
    --repeats=2 \
    --gradient_checkpointing \
    --type="t2v"
}

# ------------------------ sdv14 twoloss --------------------------------
STEP=1000
DATETIME=20251012
TRAINER="ti"
EXTRA="-1"
MODEL_PATH="/home/beihang/jzl/models/zai-org/CogVideoX-2b"

train safe $STEP $DATETIME $TRAINER $EXTRA $MODEL_PATH
