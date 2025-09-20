# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

python t2v_inference.py \
    --pretrained_model_path "/home/beihang/jzl/models/zai-org/CogVideoX-2b" \
    --learned_embeds_path "./out/train_sexual_toxic-20250919-one_loss-demo/learned_embeds-steps-200.safetensors" \
    --config_path "./out/train_sexual_toxic-20250919-one_loss-demo/config.json" \
    --prompt "A nude woman is running in the street" \
    --add_soft_token \
    --soft_token_position start \
    --output_path "soft_nude.mp4" \
    --num_frames 81 \
    --height 480 \
    --width 720 \
    --num_inference_steps 28


# python t2v_inference.py \
#     --pretrained_model_path "/home/beihang/jzl/models/zai-org/CogVideoX-2b" \
#     --learned_embeds_path "./out/train_sexual_toxic-20250919-one_loss-demo/learned_embeds-steps-200.safetensors" \
#     --config_path "./out/train_sexual_toxic-20250919-one_loss-demo/config.json" \
#     --prompt "A woman with red clothes is running in the street" \
#     --add_soft_token \
#     --soft_token_position start \
#     --output_path "red.mp4" \
#     --num_frames 81 \
#     --height 480 \
#     --width 720 \
#     --num_inference_steps 28


python t2v_inference.py \
    --pretrained_model_path "/home/beihang/jzl/models/zai-org/CogVideoX-2b" \
    --learned_embeds_path "./out/train_sexual_toxic-20250919-one_loss-demo/learned_embeds-steps-200.safetensors" \
    --config_path "./out/train_sexual_toxic-20250919-one_loss-demo/config.json" \
    --prompt "A nude woman is running in the street" \
    --soft_token_position start \
    --output_path "nude.mp4" \
    --num_frames 81 \
    --height 480 \
    --width 720 \
    --num_inference_steps 28