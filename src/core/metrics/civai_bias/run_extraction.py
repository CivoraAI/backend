#!/usr/bin/env python3
"""
Run the full extraction pipeline on news_data.json

Usage: python run_extraction.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from civai_bias.extraction import cluster_claims_by_group, write_fact_bank

def main():
    data_path = "/Users/arav/Documents/GitHub/backend/src/core/metrics/civai_bias/data/news_data.json"
    
    print("="*80)
    print("EXTRACTION PIPELINE - news_data.json")
    print("="*80)
    print("\nThis will:")
    print("  1. Extract factual claims from all articles (spaCy)")
    print("  2. Generate/cache embeddings (FlagEmbedding)")
    print("  3. Cluster similar claims (cosine similarity >= 0.85)")
    print("  4. Classify clusters (core_fact, left_claim, right_claim)")
    print("  5. Select representative claims")
    print("  6. Write factbanks to news_data.json")
    print("\nEstimated time: ~10-15 minutes (first run with caching)")
    print("                ~2-3 minutes (subsequent runs with cache)")
    print()
    
    input("Press Enter to start or Ctrl+C to cancel...")
    
    start = time.time()
    
    # Extract and cluster with caching
    print("\n[1/2] Extracting and clustering claims...")
    all_groups = cluster_claims_by_group(
        data_path,
        skip_extraction=False,  # Extract claims from articles
        use_cache=True          # Use embedding cache
    )
    
    cluster_time = time.time() - start
    print(f"\n✅ Clustering completed in {cluster_time/60:.1f} minutes")
    print(f"   Groups processed: {len(all_groups)}")
    
    # Build factbanks
    print("\n[2/2] Building factbanks...")
    factbanks = write_fact_bank(data_path, all_groups)
    
    total_time = time.time() - start
    
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Factbanks created: {len(factbanks)}")
    
    # Show statistics
    if factbanks:
        total_core = sum(len(fb.get('core_facts', [])) for fb in factbanks)
        total_left = sum(len(fb.get('claims_left', [])) for fb in factbanks)
        total_right = sum(len(fb.get('claims_right', [])) for fb in factbanks)
        
        print(f"\nFact Summary:")
        print(f"  Core facts:   {total_core} (cross-partisan)")
        print(f"  Left claims:  {total_left} (left-only)")
        print(f"  Right claims: {total_right} (right-only)")
        
        # Show percentage of core facts
        total_all = total_core + total_left + total_right
        if total_all > 0:
            core_pct = (total_core / total_all) * 100
            print(f"\n  Core facts are {core_pct:.1f}% of all claims")
    
    print(f"\n✅ Results saved to: {data_path}")
    print(f"✅ Embedding cache saved to: data/embeddings_cache.npz")

if __name__ == "__main__":
    main()

