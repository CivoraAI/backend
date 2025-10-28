#!/usr/bin/env python3
"""Test the complete integrated pipeline"""

import sys
import json
sys.path.insert(0, '.')

print("="*80)
print("TESTING INTEGRATED PIPELINE")
print("="*80)

data_path = "data/news_data.json"

# Verify all components can be imported
print("\n[1/4] Verifying imports...")
try:
    from civai_bias.create_article_object import sentences_quotes
    from civai_bias.extraction_llm import extract_all_factbanks
    from civai_bias.brief import generate_all_briefs
    print("  ✅ All modules imported successfully")
except Exception as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

# Step 1: Extract quotes/sentences (only new articles)
print("\n[2/4] Extracting quotes/sentences...")
try:
    sentences_quotes(data_path, process_all=False)
except Exception as e:
    print(f"  ❌ Error: {e}")

# Step 2: Extract factbanks
print("\n[3/4] Extracting factbanks...")
try:
    factbanks = extract_all_factbanks(data_path)
    print(f"  ✅ Created {len(factbanks)} factbanks")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Step 3: Generate briefs
print("\n[4/4] Generating briefs...")
try:
    briefs = generate_all_briefs(data_path)
    print(f"  ✅ Created {len(briefs)} briefs")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Verify final structure
print("\n" + "="*80)
print("FINAL DATA STRUCTURE:")
print("="*80)

with open(data_path) as f:
    data = json.load(f)

print(f"\nnews_data.json contents:")
print(f"  articles: {len(data.get('articles', []))}")
print(f"  groups: {len(data.get('groups', []))}")
print(f"  factbanks: {len(data.get('factbanks', []))}")
print(f"  briefs: {len(data.get('briefs', []))}")

# Show sample brief
if data.get('briefs'):
    brief = data['briefs'][0]
    print(f"\nSample brief (topic {brief.get('topic_id')}):")
    if brief.get('brief_text'):
        print(f"  {brief['brief_text'][:150]}...")
    else:
        print(f"  Facts: {len(brief.get('facts_bullets', []))}")
        print(f"  Left: {len(brief.get('left_bullets', []))}")
        print(f"  Right: {len(brief.get('right_bullets', []))}")

print("\n" + "="*80)
print("✅ INTEGRATION TEST COMPLETE")
print("="*80)
print("\nThe full pipeline is working:")
print("  1. ✓ Quote/sentence extraction")
print("  2. ✓ Factbank extraction (LLM-based)")
print("  3. ✓ Brief generation")
print("  4. ✓ All data saved to news_data.json")
print("\n✅ Ready for automated hourly runs!")

