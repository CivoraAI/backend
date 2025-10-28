#!/usr/bin/env python3
"""Run the complete pipeline: scraper + grouper + extraction"""

import sys
import os
import json
sys.path.insert(0, '.')

print("="*80)
print("FULL PIPELINE: Scraper → Grouper → Extraction")
print("="*80)

# Step 1: Check if we need to scrape
with open('data/news_data.json') as f:
    data = json.load(f)

if len(data.get('articles', [])) == 0:
    print("\n[1/3] Scraping articles...")
    from civai_bias.scraper import main
    # Run scraper once
    import subprocess
    subprocess.run([sys.executable, '-c', '''
import sys
sys.path.insert(0, ".")
from civai_bias.scraper import main
main()
'''], input='1\n'.encode())
else:
    print(f"\n[1/3] Skipping scrape - {len(data['articles'])} articles already exist")

# Step 2: Run grouper if needed
print("\n[2/3] Grouping articles...")
if len(data.get('groups', [])) == 0 or True:  # Always re-run for testing
    from civai_bias.grouper import add_groups_to_articles
    add_groups_to_articles('data/news_data.json', threshold=0.80, allow_multi_group=True)
    
    with open('data/news_data.json') as f:
        data = json.load(f)
    print(f"✅ Created {len(data.get('groups', []))} groups")
else:
    print(f"✅ Using existing {len(data.get('groups', []))} groups")

# Step 3: Run extraction
print("\n[3/3] Extracting and clustering claims...")
from civai_bias.extraction import cluster_claims_by_group, write_fact_bank

all_groups = cluster_claims_by_group(
    'data/news_data.json',
    skip_extraction=False,
    use_cache=False,  # Start fresh for testing
    topic_relevance_threshold=0.60
)

factbanks = write_fact_bank('data/news_data.json', all_groups)

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE!")
print("="*80)
with open('data/news_data.json') as f:
    data = json.load(f)
    
print(f"\nArticles: {len(data.get('articles', []))}")
print(f"Groups: {len(data.get('groups', []))}")
print(f"Factbanks: {len(data.get('factbanks', []))}")

if data.get('factbanks'):
    total_core = sum(len(fb.get('core_facts', [])) for fb in data['factbanks'])
    total_left = sum(len(fb.get('claims_left', [])) for fb in data['factbanks'])
    total_right = sum(len(fb.get('claims_right', [])) for fb in data['factbanks'])
    
    print(f"\nClaim totals:")
    print(f"  Core facts: {total_core}")
    print(f"  Left claims: {total_left}")
    print(f"  Right claims: {total_right}")

print(f"\n✅ All data saved to data/news_data.json")

