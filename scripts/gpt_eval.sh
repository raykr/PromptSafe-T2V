# export OPENAI_API_KEY="your_key"

python gpt_t2v_eval.py \
  --baseline_dir /home/raykr/projects/PromptSafe-T2V/out/tiny/sexual/baseline \
  --defense_dir  /home/raykr/projects/PromptSafe-T2V/out/tiny/sexual/multi_defense \
  --out_csv /home/raykr/projects/PromptSafe-T2V/out/tiny/sexual/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano

python gpt_t2v_eval.py \
  --baseline_dir /home/raykr/projects/PromptSafe-T2V/out/tiny/violent/baseline \
  --defense_dir  /home/raykr/projects/PromptSafe-T2V/out/tiny/violent/multi_defense \
  --out_csv /home/raykr/projects/PromptSafe-T2V/out/tiny/violent/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano

python gpt_t2v_eval.py \
  --baseline_dir /home/raykr/projects/PromptSafe-T2V/out/tiny/political/baseline \
  --defense_dir  /home/raykr/projects/PromptSafe-T2V/out/tiny/political/multi_defense \
  --out_csv /home/raykr/projects/PromptSafe-T2V/out/tiny/political/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano

python gpt_t2v_eval.py \
  --baseline_dir /home/raykr/projects/PromptSafe-T2V/out/tiny/disturbing/baseline \
  --defense_dir  /home/raykr/projects/PromptSafe-T2V/out/tiny/disturbing/multi_defense \
  --out_csv /home/raykr/projects/PromptSafe-T2V/out/tiny/disturbing/gpt_eval_results.csv \
  --fps 1 \
  --max_frames 16 \
  --model gpt-5-nano