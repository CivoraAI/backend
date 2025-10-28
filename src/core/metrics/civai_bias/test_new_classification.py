#!/usr/bin/env python3
"""Test the new sophisticated type_of_claim logic"""

import sys
sys.path.insert(0, '.')

from civai_bias.extraction import type_of_claim, load_outlet_lean

outlet_lean = load_outlet_lean()

# Test cases
test_cases = [
    {
        "name": "Diverse (4 left, 2 right)",
        "cluster": [
            {"source": "msnbc.com"},
            {"source": "cnn.com"},
            {"source": "nytimes.com"},
            {"source": "washingtonpost.com"},
            {"source": "foxnews.com"},
            {"source": "nypost.com"}
        ]
    },
    {
        "name": "Heavily left, NO right (5 left)",
        "cluster": [
            {"source": "msnbc.com"},
            {"source": "cnn.com"},
            {"source": "nytimes.com"},
            {"source": "vox.com"},
            {"source": "theatlantic.com"}
        ]
    },
    {
        "name": "Heavily right, NO left (5 right)",
        "cluster": [
            {"source": "foxnews.com"},
            {"source": "dailywire.com"},
            {"source": "nypost.com"},
            {"source": "washingtontimes.com"},
            {"source": "theepochtimes.com"}
        ]
    },
    {
        "name": "Moderate left with neutral (3 left, 2 neutral)",
        "cluster": [
            {"source": "cnn.com"},
            {"source": "nbcnews.com"},
            {"source": "npr.org"},
            {"source": "reuters.com"},
            {"source": "bbc.com"}
        ]
    },
    {
        "name": "Mixed with neutral (2 left, 1 right, 2 neutral)",
        "cluster": [
            {"source": "cnn.com"},
            {"source": "nytimes.com"},
            {"source": "foxnews.com"},
            {"source": "reuters.com"},
            {"source": "apnews.com"}
        ]
    },
    {
        "name": "Small cluster (2 claims)",
        "cluster": [
            {"source": "cnn.com"},
            {"source": "foxnews.com"}
        ]
    }
]

print("="*80)
print("NEW CLASSIFICATION LOGIC TEST")
print("="*80)
print("\nKey: CORE FACT = Has diversity OR not heavily partisan")
print("     PARTISAN = 80%+ one side AND no opposing voices\n")

for test in test_cases:
    result = type_of_claim(test["cluster"], outlet_lean)
    
    # Count sources
    left = sum(1 for c in test["cluster"] if outlet_lean.get(c["source"], 0) < 0)
    right = sum(1 for c in test["cluster"] if outlet_lean.get(c["source"], 0) > 0)
    neutral = sum(1 for c in test["cluster"] if outlet_lean.get(c["source"], 0) == 0)
    
    print(f"Test: {test['name']}")
    print(f"  Sources: {left}L / {right}R / {neutral}N (total: {len(test['cluster'])})")
    print(f"  Result: {result}")
    print()

print("="*80)
print("EXPECTED IMPROVEMENTS:")
print("="*80)
print("✅ More core facts (diversity = truth)")
print("✅ Fewer partisan claims (only extreme cases)")
print("✅ Better handling of real-world source imbalance")

