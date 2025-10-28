#!/usr/bin/env python3
"""
LLM-Based Factbank Extraction
Let the LLM do all the heavy lifting - no embeddings, no clustering
"""

import json
import os
import requests
import numpy as np

def load_outlet_lean(outlet_lean_path: str = None) -> dict:
    """Load outlet lean scores"""
    if outlet_lean_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        outlet_lean_path = os.path.join(current_dir, "..", "data", "outlet_lean.json")
    
    with open(outlet_lean_path, 'r') as f:
        return json.load(f)

def call_llm(prompt: str, model: str = "openai/gpt-4o-mini") -> str:
    """Call LLM via OpenRouter"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    # Try loading from .env file if not in environment
    if not api_key:
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
        except:
            pass
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment or .env file")
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"API error {response.status_code}: {response.text}")

def extract_factbank_for_group(articles: list, outlet_lean: dict, group_id: str) -> dict:
    """
    Use LLM to extract factbank from a group of articles.
    
    Returns factbank with core_facts, claims_left, claims_right
    """
    
    # Separate articles by source lean
    left_articles = []
    right_articles = []
    neutral_articles = []
    
    for article in articles:
        source = article['source_domain']
        lean = outlet_lean.get(source, 0)
        
        article_summary = {
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'source': source
        }
        
        if lean < -0.1:
            left_articles.append(article_summary)
        elif lean > 0.1:
            right_articles.append(article_summary)
        else:
            neutral_articles.append(article_summary)
    
    # Build prompt for LLM
    prompt = f"""You are analyzing news coverage of a single topic from multiple sources.

LEFT-LEANING SOURCES ({len(left_articles)} articles):
"""
    
    for i, art in enumerate(left_articles[:10], 1):  # Limit to 10 to avoid token limits
        prompt += f"\n{i}. [{art['source']}] {art['title']}"
        if art['description']:
            prompt += f"\n   {art['description']}"
    
    prompt += f"""

RIGHT-LEANING SOURCES ({len(right_articles)} articles):
"""
    
    for i, art in enumerate(right_articles[:10], 1):
        prompt += f"\n{i}. [{art['source']}] {art['title']}"
        if art['description']:
            prompt += f"\n   {art['description']}"
    
    if neutral_articles:
        prompt += f"""

NEUTRAL SOURCES ({len(neutral_articles)} articles):
"""
        for i, art in enumerate(neutral_articles[:10], 1):
            prompt += f"\n{i}. [{art['source']}] {art['title']}"
            if art['description']:
                prompt += f"\n   {art['description']}"
    
    prompt += """

TASK:
Extract the key factual claims from this coverage. Return a JSON object with:

{
  "core_facts": [
    "Fact 1 that ALL sources agree on",
    "Fact 2 that ALL sources agree on"
  ],
  "claims_left": [
    {"text": "Left-specific claim", "attribution": "Speaker/Organization name"},
    {"text": "Another left claim", "attribution": "Speaker/Organization name"}
  ],
  "claims_right": [
    {"text": "Right-specific claim", "attribution": "Speaker/Organization name"},
    {"text": "Another right claim", "attribution": "Speaker/Organization name"}
  ]
}

CRITICAL RULES FOR CORE FACTS:
- Only include what ALL sources (left AND right) explicitly report
- Be STRICTLY NEUTRAL - no causality, opinions, or interpretations
- Include when/where if known (dates, locations)
- Avoid subjective words: "significant", "major", "controversial", "frontrunner"
- Remove causal language: Instead of "X caused Y" say "X happened, then Y happened"
- Example: BAD: "Trump raised tariffs in response to the ad"
           GOOD: "Trump raised tariffs after Ontario aired an anti-tariff ad"
- Each fact should be independently verifiable from multiple sources

CRITICAL RULES FOR CLAIMS:
- These ARE opinionated/interpretive - that's the point
- Attribution should be PERSON/ORGANIZATION, not domain names
  - BAD: "attribution": "abcnews.go.com"
  - GOOD: "attribution": "Ontario Premier Doug Ford"
  - GOOD: "attribution": "White House Press Secretary"
  - GOOD: "attribution": "Democratic lawmakers"
- Include the speaker in the text when relevant
  - Example: "Treasury Secretary said he anticipates extension of the program"
- Claims should be assertions, predictions, or interpretations from that side
- If one side has no unique claims, return empty array

OUTPUT FORMAT:
- Maximum 5 core facts, 3 left claims, 3 right claims
- Return ONLY valid JSON, nothing else
- Sort facts/claims by importance before returning
"""
    
    print(f"  Calling LLM...")
    try:
        result = call_llm(prompt)
        
        # Parse JSON from response
        # Sometimes LLM wraps in ```json``` blocks
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0]
        elif '```' in result:
            result = result.split('```')[1].split('```')[0]
        
        factbank = json.loads(result.strip())
        
        # Add IDs to core facts (sort alphabetically for stability)
        core_facts = factbank.get('core_facts', [])
        core_facts_sorted = sorted(core_facts) if core_facts else []
        factbank['core_facts'] = [{"id": f"F{i}", "text": fact} for i, fact in enumerate(core_facts_sorted, 1)]
        
        # Handle claims (LLM should return objects with text and attribution)
        claims_left = factbank.get('claims_left', [])
        claims_right = factbank.get('claims_right', [])
        
        # If LLM returned strings instead of objects, convert them
        if claims_left and isinstance(claims_left[0], str):
            source = left_articles[0]['source'] if left_articles else "Left-leaning source"
            claims_left = [{"text": c, "attribution": source} for c in claims_left]
        
        if claims_right and isinstance(claims_right[0], str):
            source = right_articles[0]['source'] if right_articles else "Right-leaning source"
            claims_right = [{"text": c, "attribution": source} for c in claims_right]
        
        # Sort claims alphabetically for stability and add IDs
        claims_left_sorted = sorted(claims_left, key=lambda x: x.get('text', '')) if claims_left else []
        claims_right_sorted = sorted(claims_right, key=lambda x: x.get('text', '')) if claims_right else []
        
        factbank['claims_left'] = [{"id": f"CL{i}", **claim} for i, claim in enumerate(claims_left_sorted, 1)]
        factbank['claims_right'] = [{"id": f"CR{i}", **claim} for i, claim in enumerate(claims_right_sorted, 1)]
        
        # Add notes if one side is empty
        if not factbank['claims_left'] and left_articles:
            factbank['notes_left'] = "No unique left-wing claims found; coverage aligns with core facts"
        if not factbank['claims_right'] and right_articles:
            factbank['notes_right'] = "No unique right-wing claims found; coverage aligns with core facts"
        
        return factbank
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {"core_facts": [], "claims_left": [], "claims_right": []}

def detect_duplicate_topics(groups: list, articles_lookup: dict, threshold: float = 0.85) -> dict:
    """
    Detect groups that are about the same topic and should be merged.
    Returns dict mapping group_idx -> canonical_group_idx
    """
    from FlagEmbedding import BGEM3FlagModel
    
    # Get title summaries for each group
    group_summaries = []
    for group in groups:
        articles = [articles_lookup[aid] for aid in group if aid in articles_lookup]
        titles = [a.get('title', '') for a in articles if a.get('title')]
        summary = ' '.join(titles[:3])  # Use first 3 titles
        group_summaries.append(summary)
    
    # Embed summaries
    model = BGEM3FlagModel('BAAI/bge-large-en-v1.5', use_fp16=True)
    embeddings_result = model.encode(group_summaries)
    
    if isinstance(embeddings_result, dict):
        embeddings = embeddings_result['dense_vecs']
    else:
        embeddings = embeddings_result
    
    embeddings = np.array(embeddings)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    
    # Find duplicates
    merge_map = {}  # group_idx -> canonical_idx
    for i in range(len(groups)):
        if i in merge_map:
            continue
        for j in range(i + 1, len(groups)):
            if j in merge_map:
                continue
            sim = np.dot(embeddings_norm[i], embeddings_norm[j])
            if sim >= threshold:
                merge_map[j] = i  # Merge j into i
                print(f"  → Merging group {j} into group {i} (similarity: {sim:.2f})")
    
    return merge_map

def extract_all_factbanks(data_path: str, output_path: str = None, detect_duplicates: bool = True):
    """Extract factbanks for all groups using LLM"""
    
    if output_path is None:
        output_path = data_path
    
    print("="*80)
    print("LLM-BASED FACTBANK EXTRACTION")
    print("="*80)
    
    with open(data_path) as f:
        data = json.load(f)
    
    articles_lookup = {a['article_id']: a for a in data['articles']}
    groups = data['groups']
    outlet_lean = load_outlet_lean()
    
    print(f"\nLoaded:")
    print(f"  Articles: {len(data['articles'])}")
    print(f"  Groups: {len(groups)}")
    
    # Detect and merge duplicate topics
    merge_map = {}
    if detect_duplicates:
        print(f"\nDetecting duplicate topics...")
        merge_map = detect_duplicate_topics(groups, articles_lookup)
        if merge_map:
            print(f"  Found {len(merge_map)} groups to merge")
        else:
            print(f"  No duplicates found")
    
    factbanks = []
    
    for group_idx, article_ids in enumerate(groups):
        # Skip if this group was merged into another
        if group_idx in merge_map:
            print(f"\nGroup {group_idx}: Skipped (merged into group {merge_map[group_idx]})")
            continue
        
        # Collect article IDs (including from merged groups)
        all_article_ids = list(article_ids)
        for other_idx, canonical_idx in merge_map.items():
            if canonical_idx == group_idx:
                all_article_ids.extend(groups[other_idx])
        
        if len(all_article_ids) < 5:
            print(f"\nGroup {group_idx}: Skipped (only {len(all_article_ids)} articles)")
            continue
        
        articles = [articles_lookup[aid] for aid in all_article_ids if aid in articles_lookup]
        
        merged_note = f" (merged with {[k for k, v in merge_map.items() if v == group_idx]})" if any(v == group_idx for v in merge_map.values()) else ""
        print(f"\nProcessing group {group_idx} ({len(articles)} articles){merged_note}...")
        
        factbank = extract_factbank_for_group(articles, outlet_lean, group_idx)
        factbank['topic_id'] = group_idx
        
        print(f"  ✅ Core facts: {len(factbank.get('core_facts', []))}")
        print(f"  ✅ Left claims: {len(factbank.get('claims_left', []))}")
        print(f"  ✅ Right claims: {len(factbank.get('claims_right', []))}")
        
        factbanks.append(factbank)
    
    # Save
    data['factbanks'] = factbanks
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE!")
    print(f"{'='*80}")
    
    total_core = sum(len(fb.get('core_facts', [])) for fb in factbanks)
    total_left = sum(len(fb.get('claims_left', [])) for fb in factbanks)
    total_right = sum(len(fb.get('claims_right', [])) for fb in factbanks)
    
    print(f"\nCreated {len(factbanks)} factbanks")
    print(f"  Core facts: {total_core}")
    print(f"  Left claims: {total_left}")
    print(f"  Right claims: {total_right}")
    
    print(f"\nSaved to: {output_path}")
    
    return factbanks

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/news_data.json"
    extract_all_factbanks(data_file)

