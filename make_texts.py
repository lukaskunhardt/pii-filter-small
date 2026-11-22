"""Write varied texts with invented personal information, using an open-weight model (DeepSeek V4.1 Flash, MIT) via OpenRouter.

No labels here: the teacher labels the texts later (label.py). Every text gets a language, a document type and a random set of
personal-information kinds; about 15 % ask for a text with only public or business information (hard negatives). The plan comes
from a fixed seed, so a stopped run can resume: ids already in the output are skipped.

  export OPENROUTER_API_KEY=...
  python make_texts.py write --n 24000 --out texts_raw.jsonl
  python make_texts.py dedup texts_raw.jsonl texts.jsonl       # drop texts that overlap the test samples of evaluate.py
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import threading
import time

import requests

MODEL = "deepseek/deepseek-v4.1-flash-1017"
# (code, name, weight)
LANGS = [("en", "English", 12), ("de", "German", 8), ("fr", "French", 7), ("es", "Spanish", 7), ("it", "Italian", 5), ("pt", "Portuguese", 5),
         ("nl", "Dutch", 5), ("sv", "Swedish", 3), ("da", "Danish", 3), ("no", "Norwegian", 2), ("fi", "Finnish", 2), ("pl", "Polish", 4),
         ("cs", "Czech", 2), ("ro", "Romanian", 2), ("tr", "Turkish", 3), ("ru", "Russian", 4), ("ar", "Arabic", 3), ("hi", "Hindi", 3),
         ("ja", "Japanese", 4), ("zh", "Chinese (Simplified)", 4), ("ko", "Korean", 3), ("id", "Indonesian", 3)]
REGION = {"en": "US, UK, Irish or Australian", "pt": "Portuguese or Brazilian", "es": "Spanish or Latin American", "fr": "French, Belgian or Canadian",
          "de": "German, Austrian or Swiss", "ar": "Egyptian, Saudi or Moroccan", "zh": "mainland Chinese"}
DOCS = ["personal email", "work email", "customer support ticket", "chat conversation between two people", "job application letter", "CV",
        "HR record", "rental agreement excerpt", "insurance claim", "bank transfer confirmation", "invoice", "online order confirmation",
        "hospital discharge note", "doctor's referral letter", "appointment reminder SMS", "school enrollment form", "police incident report",
        "court summons", "social media direct message", "forum post", "meeting notes", "travel booking confirmation", "hotel registration form",
        "delivery note", "IT helpdesk ticket with log excerpt", "configuration file with credentials", "code snippet with a hard-coded API key",
        "password reset email", "account statement", "tax form excerpt", "visa application", "wedding invitation", "classified ad",
        "real estate listing", "voicemail transcript", "customer database export (a few rows, CSV)", "JSON API response with user data"]
KINDS = ["a person's full name", "a username or handle", "a home or postal address", "a phone number", "an email address",
         "a date of birth or another date tied to a person", "a bank account number, IBAN or credit card number",
         "an ID, passport, social security or tax number", "a personal URL or an IP address", "a password, PIN or API key"]
SYSTEM = "You write realistic documents for testing privacy software. All people and data are invented. Answer with the document only."
PROMPT = """Write a {doc} in {lang}, {words} words. Make it look like a real document of this type from a {region} context, with natural
formatting for the type. {content} Use realistic, varied formats for the local conventions. Do not mark or highlight the personal data, do not
use placeholders like [NAME] or XXX, and do not explain anything. Write in {lang} only (code and field names may stay in English)."""
POSITIVE = "Include, invented: {kinds}. Mention some of them more than once, as people do."
NEGATIVE = ("It must contain no information about private persons: only companies, products, public institutions, well-known public figures, "
            "order or product codes, prices, opening hours and public phone numbers or websites of businesses.")


def plan(n, seed=0):
    rng = random.Random(seed)
    langs = [l for l in LANGS for _ in range(l[2])]
    out = []
    for i in range(n):
        code, name, _ = rng.choice(langs)
        neg = rng.random() < 0.15
        kinds = rng.sample(KINDS, rng.randint(2, 6))
        out.append({"id": f"t{i:06d}", "lang": code, "lang_name": name, "doc": rng.choice(DOCS), "neg": neg,
                    "kinds": [] if neg else kinds, "words": rng.choice([60, 100, 150, 220, 300])})
    return out


def prompt(p):
    content = NEGATIVE if p["neg"] else POSITIVE.format(kinds="; ".join(p["kinds"]))
    return PROMPT.format(doc=p["doc"], lang=p["lang_name"], words=p["words"], region=REGION.get(p["lang"], f"{p['lang_name']}-speaking"),
                         content=content)


def chat(system, user, tries=6):
    body = {"model": MODEL, "temperature": 0.9, "max_tokens": 1200, "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
    for t in range(tries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=300)
            r.raise_for_status()
            d = r.json()
            return d["choices"][0]["message"].get("content") or "", d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                raise
            # many workers in parallel: back off longer on rate limits
            time.sleep(min(60, 2 ** t * (4 if "429" in str(e) else 2)) + random.random() * 2)


def write(n, out, workers, limit):
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [p for p in plan(n) if p["id"] not in done][: limit or None]
    lock, fh = threading.Lock(), open(out, "a", encoding="utf-8")

    def work(p):
        text, usage = chat(SYSTEM, prompt(p))
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        with lock:
            fh.write(json.dumps({**p, "text": text, "model": MODEL, "cost": usage.get("cost")}, ensure_ascii=False) + "\n")
            fh.flush()

    with cf.ThreadPoolExecutor(workers) as ex:
        for f in cf.as_completed([ex.submit(work, p) for p in todo]):
            if f.exception():
                print("error:", repr(f.exception())[:200], flush=True)


def grams(text):
    """Hashed 10-word sequences; for text without spaces (Chinese, Japanese) also 20-character sequences."""
    t = text.lower()
    words = re.findall(r"\w+", t)
    out = {hashlib.md5(" ".join(words[i:i + 10]).encode()).digest()[:8] for i in range(len(words) - 9)}
    flat = re.sub(r"\W+", "", t)
    if len(words) < 0.05 * len(flat):
        out |= {hashlib.md5(flat[i:i + 20].encode()).digest()[:8] for i in range(len(flat) - 19)}
    return out


def dedup(src, dst):
    import evaluate
    test = set()
    for d in evaluate.load_test():
        test |= grams(d["text"])
    kept = dropped = 0
    with open(dst, "w", encoding="utf-8") as fh:
        for line in open(src, encoding="utf-8"):
            if grams(json.loads(line)["text"]) & test:
                dropped += 1
            else:
                fh.write(line)
                kept += 1
    print(json.dumps({"kept": kept, "dropped": dropped}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--n", type=int, default=24000)
    w.add_argument("--out", required=True)
    w.add_argument("--workers", type=int, default=32)
    w.add_argument("--limit", type=int, default=0, help="write only this many, for a probe")
    d = sub.add_parser("dedup")
    d.add_argument("src")
    d.add_argument("dst")
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.n, a.out, a.workers, a.limit)
    else:
        dedup(a.src, a.dst)


if __name__ == "__main__":
    main()
