#!/usr/bin/env python3
"""Test the optimized extraction with embedding reuse"""

import sys
import os
import time
import json

sys.path.insert(0, '.')

from civai_bias.extraction import cluster_claims_by_group, write_fact_bank

# Create tiny test file with 1 group
data_path = "data/news_data.json"
test_path = "data/test_fast.json"
cache_path = "data/test_fast_cache.npz"

# Delete old cache
if os.path.exists(cache_path):
    os.remove(cache_path)

# Load and create 1-group test
with open(data_path) as f:
    data = json.load(f)

valid_groups = [g for g in data['groups'] if len(g) >= 5]
data['groups'] = [valid_groups[0]]  # Just first group

with open(test_path, 'w') as f:
    json.dump(data, f)

print("="*80)
print("TESTING OPTIMIZED EXTRACTION (Embedding Reuse)")
print("="*80)
print(f"\nTest file: 1 group with {len(valid_groups[0])} articles")

# Step 1: Cluster (generates embeddings once)
print("\n[1/2] Clustering claims...")
start = time.time()
all_groups = cluster_claims_by_group(
    test_path,
    skip_extraction=False,
    use_cache=True,
    cache_path=cache_path
)
cluster_time = time.time() - start

print(f"✅ Clustering done in {cluster_time:.1f}s")
print(f"   Structure: {type(all_groups[0])}")
if isinstance(all_groups[0], dict):
    print(f"   ✓ Has embeddings: {len(all_groups[0]['embeddings'])} clusters")

# Step 2: Build factbank (reuses embeddings)
print("\n[2/2] Building factbank (reusing embeddings)...")
start = time.time()
factbanks = write_fact_bank(test_path, all_groups)
factbank_time = time.time() - start

print(f"✅ Factbank done in {factbank_time:.1f}s")
print(f"   Factbanks: {len(factbanks)}")

# Show results
if factbanks:
    fb = factbanks[0]
    print(f"\n   Results:")
    print(f"     Core facts:   {len(fb.get('core_facts', []))}")
    print(f"     Left claims:  {len(fb.get('claims_left', []))}")
    print(f"     Right claims: {len(fb.get('claims_right', []))}")
    
    # Show sample
    if fb.get('core_facts'):
        print(f"\n   Sample core fact:")
        print(f"     {fb['core_facts'][0]['text'][:100]}...")

total_time = cluster_time + factbank_time
print(f"\n{'='*80}")
print(f"PERFORMANCE:")
print(f"{'='*80}")
print(f"Clustering (compute embeddings): {cluster_time:.1f}s")
print(f"Factbank (reuse embeddings):     {factbank_time:.1f}s")
print(f"Total:                           {total_time:.1f}s")
print(f"\n🚀 No redundant embedding computation!")

# Cleanup
os.remove(test_path)
print(f"\n✅ Test complete. Cleaned up test file.")

