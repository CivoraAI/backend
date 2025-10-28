import json
import re
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def call_openrouter(prompt: str, model: str = "gpt-4o-mini") -> str:
    import re, os, requests

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 400
    }

    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"].strip()

    # 🧠 Clean up: remove meta chatter ("Let's craft", "Word count", etc.)
    lines = content.splitlines()
    # Keep only lines after the last double line break or after “draft:”
    final_text = re.split(r"(?:[Dd]raft:|Word count[: ]|Let's craft|We need to keep)", content)[-1].strip()

    # If too short, fallback to original
    if len(final_text.split()) < 40:
        final_text = content
    return final_text.strip()

def generate_brief_from_factbank(factbank: dict, use_llm: bool = True) -> dict:
    """
    Generate a brief from a factbank object.
    
    Args:
        factbank: Factbank dict with core_facts, claims_left, claims_right
        use_llm: Whether to use LLM for generation
    
    Returns:
        Brief dict with topic_id and brief_text
    """
    core = factbank.get("core_facts", [])
    left_claims = factbank.get("claims_left", [])
    right_claims = factbank.get("claims_right", [])

    # 2) clean + dedupe facts
    facts_raw = [str(x.get("text", "")).strip() for x in core if str(x.get("text", "")).strip()]
    # normalize punctuation, ensure trailing period
    facts_norm = [re.sub(r'[“”‘’]', '"', t).replace("  ", " ").rstrip(".") + "." for t in facts_raw]
    # case-insensitive de-dupe while preserving original casing
    first_seen = {}
    for t in facts_norm:
        key = t.lower().strip()
        if key and key not in first_seen:
            first_seen[key] = t
    facts = list(first_seen.values())

    # Use all facts (always true for automated pipeline)

    # 3) clean + dedupe left/right claims (keep attribution)
    def clean_pairs(pairs):
        cleaned = []
        for text, attr in pairs:
            t = str(text or "").strip()
            a = str(attr or "").strip()
            if not t:
                continue
            t = re.sub(r'[“”‘’]', '"', t).replace("  ", " ").rstrip(".") + "."
            cleaned.append((t, a))
        # de-dupe on lowercased text
        fs = {}
        for t, a in cleaned:
            k = t.lower().strip()
            if k not in fs:
                fs[k] = (t, a)
        return list(fs.values())

    left_all = clean_pairs([(c.get("text", ""), c.get("attribution", "")) for c in left_claims])
    right_all = clean_pairs([(c.get("text", ""), c.get("attribution", "")) for c in right_claims])

    # 4) build bullets (UI will render bullets; we just prep strings)
    facts_bullets = [f"- {t}" for t in facts]

    def to_bullets(pairs):
        bullets = []
        for text, attr in pairs:
            tail = f" — {attr}." if attr else "."
            bullets.append(f"- {text.rstrip('.')}{tail}")
        # fix accidental '..' if text already had period
        bullets = [b.replace("..", ".") for b in bullets]
        return bullets

    left_bullets = to_bullets(left_all)
    right_bullets = to_bullets(right_all)

    # 5) plain-text fallback (still bullets)
    sections = []
    if facts_bullets:
        sections.append("Facts:\n" + "\n".join(facts_bullets))
    if left_bullets:
        sections.append("Supporters say:\n" + "\n".join(left_bullets))
    if right_bullets:
        sections.append("Opponents say:\n" + "\n".join(right_bullets))
    neutral_bullets = "\n\n".join(sections)

    llm_brief = None
    if use_llm:
        prompt = f"""
        You are a neutral journalist AI. Summarize the material below into a clear, easy-to-read brief for Gen Z readers.

        Rules:
        - Do NOT add new facts or numbers.
        - Keep the meaning exactly the same.
        - Keep it neutral (no loaded language).
        - Present it in a smooth, casual but factual tone.
        - 120–160 words max.
        - Mention both sides briefly if given.

        Facts:
        {chr(10).join(facts_bullets)}

        The left's perspective:
        {chr(10).join(left_bullets)}

        The right's perspective:
        {chr(10).join(right_bullets)}
                """.strip()

        try:
            llm_brief = call_openrouter(prompt)
        except Exception as e:
            print(f"[WARN] LLM call failed: {e}")
            llm_brief = None
    brief_obj = {
        "topic_id": factbank.get("topic_id"),
        "brief_text": llm_brief if llm_brief else None,
        "facts_bullets": facts_bullets,
        "left_bullets": left_bullets,
        "right_bullets": right_bullets,
        "neutral_bullets": neutral_bullets
    }
    
    return brief_obj

def generate_all_briefs(data_path: str) -> list:
    """
    Generate briefs for all factbanks in news_data.json.
    Only regenerates if factbanks have changed.
    
    Returns list of briefs
    """
    import hashlib
    
    with open(data_path) as f:
        data = json.load(f)
    
    factbanks = data.get('factbanks', [])
    
    if not factbanks:
        print("  No factbanks to generate briefs for")
        return []
    
    # Check if factbanks changed
    factbank_hash = hashlib.md5(json.dumps(factbanks, sort_keys=True).encode()).hexdigest()
    last_hash = data.get('_factbank_hash')
    
    if last_hash == factbank_hash and 'briefs' in data and data['briefs']:
        print("  Factbanks unchanged, skipping brief generation")
        return data['briefs']
    
    print(f"  Generating briefs for {len(factbanks)} factbanks...")
    
    briefs = []
    for i, factbank in enumerate(factbanks):
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(factbanks)} briefs...")
        
        brief = generate_brief_from_factbank(factbank, use_llm=True)
        briefs.append(brief)
    
    # Save briefs and hash
    data['briefs'] = briefs
    data['_factbank_hash'] = factbank_hash
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"  ✅ Generated {len(briefs)} briefs")
    
    return briefs