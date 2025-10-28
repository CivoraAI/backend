#!/usr/bin/env python3
"""Check the progress of the running extraction"""

import json
import os
import time

data_path = "data/news_data.json"
cache_path = "data/embeddings_cache.npz"

print("="*80)
print("EXTRACTION PROGRESS CHECK")
print("="*80)

# Check if extraction process is running
import subprocess
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
extraction_running = 'run_extraction.py' in result.stdout

print(f"\nExtraction process: {'🟢 RUNNING' if extraction_running else '🔴 NOT RUNNING'}")

# Check cache file
if os.path.exists(cache_path):
    cache_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
    cache_mtime = time.ctime(os.path.getmtime(cache_path))
    print(f"\nEmbedding cache:")
    print(f"  Size: {cache_size:.2f} MB")
    print(f"  Last updated: {cache_mtime}")
    
    # Estimate progress
    # Rough estimate: ~1MB per 200-300 claims
    estimated_claims = cache_size * 250
    print(f"  Estimated cached claims: ~{int(estimated_claims)}")
else:
    print(f"\nEmbedding cache: Not created yet")

# Check data file
try:
    with open(data_path) as f:
        data = json.load(f)
    
    print(f"\nnews_data.json:")
    print(f"  Articles: {len(data.get('articles', []))}")
    print(f"  Groups: {len(data.get('groups', []))}")
    print(f"  Valid groups (≥5): {len([g for g in data.get('groups', []) if len(g) >= 5])}")
    
    if 'claims_by_group' in data and data['claims_by_group']:
        print(f"  Claims extracted: ✓ ({len(data['claims_by_group'])} groups)")
    else:
        print(f"  Claims extracted: Not yet")
    
    if 'factbanks' in data and data['factbanks']:
        print(f"  Factbanks: {len(data['factbanks'])}")
        
        # Check if they're new format (no BS claims)
        fb = data['factbanks'][0]
        has_bs = 'bs_claims' in fb or 'bs_facts' in fb
        
        if has_bs:
            print(f"    Format: OLD (has bs_claims)")
        else:
            print(f"    Format: NEW ✓")
            
        total_core = sum(len(fb.get('core_facts', [])) for fb in data['factbanks'])
        total_left = sum(len(fb.get('claims_left', [])) for fb in data['factbanks'])
        total_right = sum(len(fb.get('claims_right', [])) for fb in data['factbanks'])
        
        print(f"\n  Current totals:")
        print(f"    Core facts:   {total_core}")
        print(f"    Left claims:  {total_left}")
        print(f"    Right claims: {total_right}")
    else:
        print(f"  Factbanks: Not created yet")
        
except Exception as e:
    print(f"\n❌ Error reading file: {e}")

print("\n" + "="*80)
if extraction_running:
    print("⏳ Extraction still running... Check again in a few minutes")
    print("   Estimated time remaining: ~20 minutes")
else:
    print("✅ Extraction process not running")
    print("   Either completed or needs to be started")
print("="*80)

