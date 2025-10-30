#!/usr/bin/env python3
"""
Force regenerate all briefs from factbanks in news_data.json.
This bypasses the hash check and regenerates all briefs.
"""

import sys
import os

# Add the civai_bias directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from civai_bias.brief import generate_brief_from_factbank
import json
import hashlib

# Default path relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "news_data.json")

# If running from backend root, use absolute path
if not os.path.exists(DATA_PATH):
    DATA_PATH = "src/core/metrics/civai_bias/data/news_data.json"

def force_regenerate_briefs(data_path: str, force: bool = True) -> list:
    """
    Force regenerate all briefs, bypassing hash check.
    
    Args:
        data_path: Path to news_data.json
        force: If True, always regenerate (ignores hash check)
    
    Returns:
        List of generated briefs
    """
    with open(data_path) as f:
        data = json.load(f)
    
    factbanks = data.get('factbanks', [])
    
    if not factbanks:
        print("  ❌ No factbanks found in data file")
        return []
    
    print(f"  📝 Found {len(factbanks)} factbanks")
    
    if not force:
        # Check hash (normal behavior)
        factbank_hash = hashlib.md5(json.dumps(factbanks, sort_keys=True).encode()).hexdigest()
        last_hash = data.get('_factbank_hash')
        
        if last_hash == factbank_hash and 'briefs' in data and data['briefs']:
            print("  ⏭️  Factbanks unchanged, skipping brief generation")
            print("     (Use force=True to regenerate anyway)")
            return data['briefs']
    
    print(f"  🔄 Generating briefs for {len(factbanks)} factbanks...")
    print("     (This will call the LLM API, so it may take a few minutes)")
    
    briefs = []
    for i, factbank in enumerate(factbanks):
        topic_id = factbank.get('topic_id', i)
        print(f"    [{i+1}/{len(factbanks)}] Topic {topic_id}...", end=" ", flush=True)
        
        try:
            brief = generate_brief_from_factbank(factbank, use_llm=True)
            briefs.append(brief)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
            # Add error brief so structure is maintained
            briefs.append({
                "topic_id": topic_id,
                "core_facts_brief": None,
                "left_claims_brief": None,
                "right_claims_brief": None,
                "facts_bullets": [],
                "left_bullets": [],
                "right_bullets": [],
                "neutral_bullets": "",
                "error": str(e)
            })
    
    # Save briefs and hash
    data['briefs'] = briefs
    factbank_hash = hashlib.md5(json.dumps(factbanks, sort_keys=True).encode()).hexdigest()
    data['_factbank_hash'] = factbank_hash
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\n  ✅ Generated {len(briefs)} briefs and saved to {data_path}")
    
    return briefs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate briefs from factbanks")
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help=f"Path to news_data.json (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if factbanks haven't changed"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("⚠️  WARNING: OPENROUTER_API_KEY not set in environment")
        print("   Briefs will be generated without LLM enhancement")
    
    # Run regeneration
    briefs = force_regenerate_briefs(args.data_path, force=args.force)
    
    if briefs:
        print(f"\n✅ Successfully regenerated {len(briefs)} briefs!")
    else:
        print("\n❌ No briefs were generated")

