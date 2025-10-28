#!/usr/bin/env python3
"""
Simplified Factbank Extraction
Uses article descriptions + centroid-based summarization
"""

import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import os
import requests

def load_outlet_lean(outlet_lean_path: str = None) -> dict:
    """Load outlet lean scores"""
    if outlet_lean_path is None:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        outlet_lean_path = os.path.join(current_dir, "..", "data", "outlet_lean.json")
    
    with open(outlet_lean_path, 'r') as f:
        return json.load(f)

def get_lead_sentences(text: str, n: int = 2) -> str:
    """Extract first N sentences from text"""
    import re
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+\s+', text)
    lead = '. '.join(sentences[:n])
    if lead and not lead.endswith('.'):
        lead += '.'
    return lead.strip()

def deflame_text(text: str, claim_type: str = "core") -> str:
    """
    Use LLM to remove inflammatory language and make text neutral.
    
    Args:
        text: The text to deflame
        claim_type: "core", "left", or "right" - determines level of neutralization
    """
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return text  # Skip deflaming if no API key
        
        if claim_type == "core":
            prompt = f"""Rewrite this text to be completely neutral, factual, and objective. Remove ALL:
- Inflammatory words
- Emotional language
- Biased framing
- Value judgments
- Partisan language

Make it sound like it was written by a neutral encyclopedia. Keep only the core facts:

"{text}"

Return ONLY the rewritten neutral text, nothing else."""
        else:
            # For left/right claims, keep some perspective but remove extreme language
            prompt = f"""Tone down inflammatory language in this text while preserving the perspective and main point. Remove extreme modifiers but keep the general viewpoint:

"{text}"

Return ONLY the toned-down text, nothing else."""
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a neutral text editor that removes inflammatory language."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            deflamed = result['choices'][0]['message']['content'].strip()
            # Remove quotes if LLM added them
            if deflamed.startswith('"') and deflamed.endswith('"'):
                deflamed = deflamed[1:-1]
            return deflamed
        else:
            print(f"  Warning: API error {response.status_code}")
            return text
            
    except Exception as e:
        print(f"  Warning: Failed to deflame text: {e}")
        return text  # Return original if deflaming fails

def select_all_descriptions_ranked(descriptions: list, embeddings: np.ndarray, sources: list, 
                                   outlet_lean: dict, max_claims: int = 5) -> list:
    """
    Select multiple descriptions ranked by centrality and source diversity.
    Returns list of (description, source) tuples.
    """
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    
    # Compute centroid
    centroid = np.mean(embeddings_norm, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    
    # Rank by similarity to centroid
    similarities = embeddings_norm @ centroid_norm
    
    # Create list of (similarity, description, source, lean)
    ranked = []
    for i, (sim, desc, source) in enumerate(zip(similarities, descriptions, sources)):
        lean = outlet_lean.get(source, 0)
        ranked.append((sim, desc, source, lean))
    
    # Sort by similarity (descending)
    ranked.sort(reverse=True, key=lambda x: x[0])
    
    # Select top descriptions with diversity
    selected = []
    seen_leans = set()
    
    for sim, desc, source, lean in ranked:
        if len(selected) >= max_claims:
            break
        
        # Prefer descriptions from diverse sources
        lean_category = 'left' if lean < -0.1 else 'right' if lean > 0.1 else 'neutral'
        
        # Always add if we haven't seen this lean category yet, or if it's high similarity
        if lean_category not in seen_leans or sim > 0.7:
            selected.append((desc, source))
            seen_leans.add(lean_category)
    
    return selected

def extract_factbanks_simple(data_path: str, output_path: str = None):
    """
    Simple factbank extraction using article descriptions.
    
    Process:
    1. For each group, get all article descriptions
    2. Embed descriptions
    3. Find centroid → select representative description
    4. Classify by source diversity
    """
    if output_path is None:
        output_path = data_path
    
    print("="*80)
    print("SIMPLIFIED FACTBANK EXTRACTION")
    print("="*80)
    
    # Load data
    with open(data_path) as f:
        data = json.load(f)
    
    articles = {a['article_id']: a for a in data['articles']}
    groups = data['groups']
    outlet_lean = load_outlet_lean()
    
    print(f"\nLoaded:")
    print(f"  Articles: {len(articles)}")
    print(f"  Groups: {len(groups)}")
    
    # Load embedding model
    print(f"\nLoading embedding model...")
    model = BGEM3FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
    
    factbanks = []
    
    for group_idx, article_ids in enumerate(groups):
        if len(article_ids) < 5:  # Skip small groups
            continue
        
        print(f"\nProcessing group {group_idx} ({len(article_ids)} articles)...")
        
        # Get articles and their descriptions
        group_articles = [articles[aid] for aid in article_ids if aid in articles]
        descriptions = []
        sources = []
        
        for article in group_articles:
            # Use description, or fall back to first 2 sentences
            desc = article.get('description', '')
            if not desc or len(desc) < 50:
                desc = get_lead_sentences(article.get('full_text', ''), n=2)
            
            if desc and len(desc) >= 50:  # Only use substantial descriptions
                descriptions.append(desc)
                sources.append(article['source_domain'])
        
        if len(descriptions) < 3:
            print(f"  Skipped: not enough descriptions")
            continue
        
        # Embed descriptions
        embeddings_result = model.encode(descriptions)
        
        # Handle both dict and array returns from model
        if isinstance(embeddings_result, dict):
            embeddings = embeddings_result['dense_vecs']
        else:
            embeddings = embeddings_result
        
        embeddings = np.array(embeddings)
        
        # Analyze source diversity
        left_count = sum(1 for s in sources if outlet_lean.get(s, 0) < -0.1)
        right_count = sum(1 for s in sources if outlet_lean.get(s, 0) > 0.1)
        neutral_count = len(sources) - left_count - right_count
        
        # Stricter diversity: need at least 2 from each side, not just 1
        has_diversity = (left_count >= 2 and right_count >= 2)
        
        print(f"  Sources: {left_count}L, {right_count}R, {neutral_count}N")
        print(f"  Diversity: {'Yes' if has_diversity else 'No'}")
        
        # Build factbank
        factbank = {
            "topic_id": f"topic_{group_idx}",
            "core_facts": [],
            "claims_left": [],
            "claims_right": []
        }
        
        if has_diversity:
            # Get multiple claims from diverse sources
            selected_claims = select_all_descriptions_ranked(
                descriptions, embeddings, sources, outlet_lean, max_claims=5
            )
            
            print(f"  Core facts: {len(selected_claims)} claims")
            seen_texts = set()
            fact_num = 1
            
            for desc, source in selected_claims:
                # Deflame for core facts (complete neutralization)
                deflamed = deflame_text(desc, claim_type="core")
                
                # Deduplicate: skip if very similar to existing fact
                if deflamed.lower() in seen_texts:
                    print(f"    Skipped duplicate")
                    continue
                    
                seen_texts.add(deflamed.lower())
                factbank["core_facts"].append({
                    "id": f"F{fact_num}",
                    "text": deflamed
                })
                print(f"    F{fact_num}: {deflamed[:70]}...")
                fact_num += 1
        else:
            # Partisan group - create left or right claims
            selected_claims = select_all_descriptions_ranked(
                descriptions, embeddings, sources, outlet_lean, max_claims=3
            )
            
            if left_count > right_count:
                print(f"  Left claims: {len(selected_claims)} claims")
                seen_texts = set()
                claim_num = 1
                
                for desc, source in selected_claims:
                    # Tone down but keep perspective
                    deflamed = deflame_text(desc, claim_type="left")
                    
                    # Deduplicate
                    if deflamed.lower() in seen_texts:
                        print(f"    Skipped duplicate")
                        continue
                    
                    seen_texts.add(deflamed.lower())
                    factbank["claims_left"].append({
                        "id": f"CL{claim_num}",
                        "text": deflamed,
                        "attribution": source
                    })
                    print(f"    CL{claim_num}: {deflamed[:70]}...")
                    claim_num += 1
                    
            elif right_count > left_count:
                print(f"  Right claims: {len(selected_claims)} claims")
                seen_texts = set()
                claim_num = 1
                
                for desc, source in selected_claims:
                    # Tone down but keep perspective
                    deflamed = deflame_text(desc, claim_type="right")
                    
                    # Deduplicate
                    if deflamed.lower() in seen_texts:
                        print(f"    Skipped duplicate")
                        continue
                    
                    seen_texts.add(deflamed.lower())
                    factbank["claims_right"].append({
                        "id": f"CR{claim_num}",
                        "text": deflamed,
                        "attribution": source
                    })
                    print(f"    CR{claim_num}: {deflamed[:70]}...")
                    claim_num += 1
        
        factbanks.append(factbank)
    
    # Save factbanks
    data['factbanks'] = factbanks
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"Created {len(factbanks)} factbanks")
    
    total_core = sum(len(fb['core_facts']) for fb in factbanks)
    total_left = sum(len(fb['claims_left']) for fb in factbanks)
    total_right = sum(len(fb['claims_right']) for fb in factbanks)
    
    print(f"\nTotals:")
    print(f"  Core facts: {total_core}")
    print(f"  Left claims: {total_left}")
    print(f"  Right claims: {total_right}")
    
    print(f"\nSaved to: {output_path}")
    
    return factbanks


if __name__ == "__main__":
    import sys
    
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/news_data.json"
    extract_factbanks_simple(data_file)

