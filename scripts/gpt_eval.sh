# export OPENAI_API_KEY="your_key"

python gpt_t2v_eval.py \
  --baseline_dir /home/raykr/projects/PromptSafe-T2V/out/toxic/baseline \
  --defense_dir  /home/raykr/projects/PromptSafe-T2V/out/toxic/multi_defense \
  --out_csv /home/raykr/projects/PromptSafe-T2V/out/toxic/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano
