#!/usr/bin/env python3
"""
Brief Generator for All Factbanks
Wrapper around brief.py to process all factbanks from news_data.json
"""

import json
import os
from civai_bias.brief import generate_brief

def generate_all_briefs(data_path: str) -> list:
    """
    Generate briefs for all factbanks in news_data.json
    
    Returns list of briefs
    """
    print("Generating briefs for all factbanks...")
    
    with open(data_path) as f:
        data = json.load(f)
    
    factbanks = data.get('factbanks', [])
    
    if not factbanks:
        print("  No factbanks found!")
        return []
    
    briefs = []
    
    # Create temp directory for factbank files
    temp_dir = os.path.join(os.path.dirname(data_path), 'temp_factbanks')
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, factbank in enumerate(factbanks):
        topic_id = factbank.get('topic_id', i)
        print(f"  Generating brief for topic {topic_id}...")
        
        # Save factbank to temp file
        temp_factbank_path = os.path.join(temp_dir, f'factbank_{topic_id}.json')
        with open(temp_factbank_path, 'w') as f:
            json.dump(factbank, f, indent=2)
        
        try:
            # Generate brief
            brief_result = generate_brief(temp_factbank_path, use_all_facts=True, use_llm=True)
            
            # Add topic_id to result
            brief_result['topic_id'] = topic_id
            briefs.append(brief_result)
            
            print(f"    ✅ Brief generated ({len(brief_result.get('brief', '').split())} words)")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            briefs.append({
                'topic_id': topic_id,
                'brief': '',
                'error': str(e)
            })
        
        # Clean up temp file
        os.remove(temp_factbank_path)
    
    # Clean up temp directory
    os.rmdir(temp_dir)
    
    # Save briefs to data file
    data['briefs'] = briefs
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\n✅ Generated {len(briefs)} briefs")
    return briefs

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/news_data.json"
    generate_all_briefs(data_file)

