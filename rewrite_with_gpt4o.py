#!/usr/bin/env python3
import os
import csv
import time
import argparse
from typing import List

try:
    # OpenAI SDK v1 style
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are a safety rewriter for text-to-video (T2V) prompts.\n"
    "Goal: Remove unsafe/inappropriate/explicit/illegal/violent/hate/harassment/adult/gory elements while preserving the original intent and scene coherence as much as possible.\n"
    "Requirements:\n"
    "1) Preserve temporal sequencing and shot progression (e.g., start→process→end, before-after relations, continuous actions, shot transitions).\n"
    "2) Keep neutral subjects and actions (people, places, weather, object interactions), replacing any unsafe elements with healthy and compliant expressions.\n"
    "3) Do NOT include minors, explicit sexual acts, sexual organs, sexual insinuation, violence, hate, discrimination, self-harm, drugs, crimes, or illegal content.\n"
    "4) Write in concise, clear English; multiple sentences are allowed to reflect the temporal nature.\n"
    "5) Do not change the core narrative goal/theme (e.g., keep the same location, characters, and time flow, while removing unsafe details).\n"
    "Output: Only the rewritten safe prompt text. Do not add any extra explanations."
)


def read_prompts_from_txt(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def create_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Please install: pip install 'openai>=1.0.0'")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not found")
    return OpenAI(api_key=api_key)


def rewrite_prompt(client: "OpenAI", prompt: str, model: str = "gpt-4o", max_retries: int = 5, backoff: float = 1.5) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content.strip()
            return content
        except Exception as e:  # APIError, RateLimitError, etc.
            last_err = e
            sleep_s = (backoff ** attempt)
            time.sleep(min(30, sleep_s))
    raise RuntimeError(f"改写失败：{last_err}")


def write_csv(output_csv: str, rows: List[List[str]]):
    # 确保目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    header = ["label", "prompt", "rewritten_prompt", "malicious_video_path", "safe_video_path"]
    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Rewrite unsafe T2V prompts into safe ones using GPT-4o and save to CSV")
    parser.add_argument("--input_txt", type=str, required=True, help="Path to input TXT file, e.g., datasets/Tiny-T2VSafetyBench/1.txt")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file, e.g., datasets/train/caption.csv")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name, default gpt-4o")
    parser.add_argument("--label", type=str, default="toxic", help="CSV label column value, default toxic")
    parser.add_argument("--malicious_video_dir", type=str, default="videos", help="Placeholder directory for malicious video path")
    parser.add_argument("--safe_video_dir", type=str, default="videos", help="Placeholder directory for safe video path")
    parser.add_argument("--dry_run", action="store_true", help="Print only, do not write CSV or call API")

    args = parser.parse_args()

    prompts = read_prompts_from_txt(args.input_txt)
    client = None
    if not args.dry_run:
        client = create_client()

    rows: List[List[str]] = []
    for idx, p in enumerate(prompts):
        if args.dry_run:
            # Dry run: do not call API, provide a placeholder safe rewrite sample
            safe = f"[Safe rewrite sample] {p[:120]}"
        else:
            assert client is not None
            safe = rewrite_prompt(client, p, model=args.model)
        malicious_video_path = os.path.join(args.malicious_video_dir, f"malicious_{idx:05d}.mp4")
        safe_video_path = os.path.join(args.safe_video_dir, f"safe_{idx:05d}.mp4")
        rows.append([args.label, p, safe, malicious_video_path, safe_video_path])

    if args.dry_run:
        for r in rows:
            print(r)
    else:
        write_csv(args.output_csv, rows)
        print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()


