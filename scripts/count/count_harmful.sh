# python analyze_eval_results.py \
#     --csv /home/raykr/projects/PromptSafe-T2V/out/tiny/sexual/gpt_eval_results.csv \
#     --output /home/raykr/projects/PromptSafe-T2V/out/tiny/sexual/eval_statistics.txt

# python analyze_eval_results.py \
#     --csv /home/raykr/projects/PromptSafe-T2V/out/tiny/violent/gpt_eval_results.csv \
#     --output /home/raykr/projects/PromptSafe-T2V/out/tiny/violent/eval_statistics.txt

# python analyze_eval_results.py \
#     --csv /home/raykr/projects/PromptSafe-T2V/out/tiny/political/gpt_eval_results.csv \
#     --output /home/raykr/projects/PromptSafe-T2V/out/tiny/political/eval_statistics.txt

# python analyze_eval_results.py \
#     --csv /home/raykr/projects/PromptSafe-T2V/out/tiny/disturbing/gpt_eval_results.csv \
#     --output /home/raykr/projects/PromptSafe-T2V/out/tiny/disturbing/eval_statistics.txt

python analyze_eval_results.py \
    --csv out/CogVideoX-5b/tiny/sexual/gpt_eval_results.csv \
    --output out/CogVideoX-5b/tiny/sexual/eval_statistics.txt

python analyze_eval_results.py \
    --csv out/CogVideoX-5b/tiny/violent/gpt_eval_results.csv \
    --output out/CogVideoX-5b/tiny/violent/eval_statistics.txt

python analyze_eval_results.py \
    --csv out/CogVideoX-5b/tiny/political/gpt_eval_results.csv \
    --output out/CogVideoX-5b/tiny/political/eval_statistics.txt

python analyze_eval_results.py \
    --csv out/CogVideoX-5b/tiny/disturbing/gpt_eval_results.csv \
    --output out/CogVideoX-5b/tiny/disturbing/eval_statistics.txt