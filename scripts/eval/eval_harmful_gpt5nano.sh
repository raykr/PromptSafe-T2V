# export OPENAI_API_KEY="your_key"

python gpt_t2v_eval.py \
  --baseline_dir out/CogVideoX1.5-5B/tiny/sexual/baseline \
  --defense_dir  out/CogVideoX1.5-5B/tiny/sexual/multi_defense \
  --out_csv out/CogVideoX1.5-5B/tiny/sexual/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano

python gpt_t2v_eval.py \
  --baseline_dir out/CogVideoX1.5-5B/tiny/violent/baseline \
  --defense_dir  out/CogVideoX1.5-5B/tiny/violent/multi_defense \
  --out_csv out/CogVideoX1.5-5B/tiny/violent/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano

python gpt_t2v_eval.py \
  --baseline_dir out/CogVideoX1.5-5B/tiny/political/baseline \
  --defense_dir  out/CogVideoX1.5-5B/tiny/political/multi_defense \
  --out_csv out/CogVideoX1.5-5B/tiny/political/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano

python gpt_t2v_eval.py \
  --baseline_dir out/CogVideoX1.5-5B/tiny/disturbing/baseline \
  --defense_dir  out/CogVideoX1.5-5B/tiny/disturbing/multi_defense \
  --out_csv out/CogVideoX1.5-5B/tiny/disturbing/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano