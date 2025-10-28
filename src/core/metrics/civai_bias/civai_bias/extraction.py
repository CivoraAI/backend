import spacy
import json
import warnings
import os
import sys
import subprocess
import re
from ast import Pass
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from .metrics import load_sentiment_lexicon
from . import rules

# Suppress warnings
warnings.filterwarnings("ignore", message=".*similarity.*")


# ============================================================================
# CONFIGURATION
# ============================================================================

CORRECT_PYTHON = "/opt/anaconda3/envs/civai_py310/bin/python"


# ============================================================================
# SETUP: Python Environment & Model Loading
# ============================================================================

def check_and_switch_python():
    """Check if we're using the correct Python interpreter and switch if needed"""
    if sys.executable != CORRECT_PYTHON:
        print("Wrong Python interpreter detected, switching...")

        # Set environment variables for the subprocess
        env = os.environ.copy()
        env.update(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )

        try:
            # Re-run this script with the correct Python interpreter
            _ = subprocess.run(
                [CORRECT_PYTHON, __file__],
                env=env,
                cwd=os.path.dirname(__file__),  # Stay in iter_4 directory,
                check=True,
            )
            
            sys.exit(0)  # Exit the current process since we've delegated to the correct one
        except subprocess.CalledProcessError as e:
            print(f"Error running with correct Python: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Python interpreter not found: {CORRECT_PYTHON}")
            print("Please check your Anaconda installation")
            print("Continuing with current Python interpreter...")
    else:
        Pass()

def load_article_bank(article_bank_path):
    with open(article_bank_path, 'r') as file:
        data = json.load(file)
    
    # Handle both formats: {"articles": [...]} or [...]
    if isinstance(data, dict) and "articles" in data:
        return data
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unexpected JSON format. Expected list or dict with 'articles' key.")

def load_embedding_model():
    check_and_switch_python()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU only
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from FlagEmbedding import FlagModel
    return FlagModel(
        "BAAI/bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        devices="cpu",  # Use CPU by default, change to "cuda:0" if GPU available
        pooling_method="cls")

def load_spacy_model():
    check_and_switch_python()
    try:
        nlp = spacy.load("en_core_web_lg")
        print("✅ Loaded en_core_web_lg model")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
            print("✅ Loaded en_core_web_md model")
        except OSError:
            try:
                nlp = spacy.load("en")
                print("✅ Loaded basic English model")
            except OSError:
                print("❌ No spaCy English model found. Please install with:")
                print("python -m spacy download en_core_web_sm")
                exit(1)
    return nlp


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize article text"""
    # Normalize quotes/apos
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove boilerplate
    text = re.sub(r'Photo:.*?\|', '', text)
    text = re.sub(r'By .*?\|', '', text)
    # Keep paragraph boundaries
    text = re.sub(r'\n+', '\n', text)
    # Remove any abnormal characters or symbols
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()


# ============================================================================
# CLAIM FILTERING: Remove noise and off-topic content
# ============================================================================

def is_noise_claim(claim_text: str) -> bool:
    """
    Filter out common noise patterns that aren't real claims.
    
    Returns True if claim should be DISCARDED.
    """
    text_lower = claim_text.lower()
    
    # Navigation/UI text
    navigation_patterns = [
        "download", "subscribe", "sign up", "newsletter", "click here",
        "read more", "related stories", "breaking news", "top stories",
        "share this", "follow us", "enter your email", "sign me up"
    ]
    if any(pattern in text_lower for pattern in navigation_patterns):
        return True
    
    # Attribution lines (reporter bylines)
    attribution_patterns = [
        "is a reporter", "is a correspondent", "is an editor",
        "is a senior", "is a contributor", "is a writer",
        "contributed to this report", "contributed reporting"
    ]
    if any(pattern in text_lower for pattern in attribution_patterns):
        return True
    
    # Image captions / metadata
    if claim_text.startswith(("Photo:", "Image:", "File/", "Getty Images", "AP/", "hide caption")):
        return True
    
    # Very short or incomplete
    if len(claim_text.strip()) < 25:
        return True
    
    # All caps (usually headers)
    if claim_text.isupper():
        return True
    
    return False

def is_background_context(claim_text: str, claim_obj: dict) -> bool:
    """
    Detect historical background or tangential context.
    
    Returns True if claim should be DISCARDED.
    """
    import re
    
    # Check for old dates in the text itself (not just extracted dates)
    year_matches = re.findall(r'\b(19\d{2}|20[0-1]\d)\b', claim_text)
    for year_str in year_matches:
        year = int(year_str)
        if year < 2020:
            return True
    
    # Also check extracted dates
    dates = claim_obj.get("dates", [])
    for date in dates:
        year_match = re.search(r'\b(19\d{2}|20[0-1]\d)\b', date)
        if year_match:
            year = int(year_match.group(1))
            if year < 2020:
                return True
    
    # Generic procedural/legal language
    procedural_phrases = [
        "legal opinion", "according to law", "under the law",
        "historically", "in the past", "over the years",
        "traditionally", "has been held for many years"
    ]
    text_lower = claim_text.lower()
    if any(phrase in text_lower for phrase in procedural_phrases):
        return True
    
    return False

def is_political_tangent(claim_text: str) -> bool:
    """
    Detect when a claim is a political tangent/side story.
    
    Common tangents in political news:
    - Shutdown/budget fights (when main story is something else)
    - Electoral politics (when main story is policy)
    - Investigative/legal proceedings (when main story is different)
    - Poll numbers/approval ratings
    
    Returns True if claim should be DISCARDED.
    """
    text_lower = claim_text.lower()
    
    # Shutdown/budget tangents
    shutdown_patterns = [
        "government shutdown", "shutdown", "blocked a bill",
        "blocked an effort", "blocked legislation",
        "pay federal workers", "spending bill", "budget fight"
    ]
    
    # Electoral tangents
    electoral_patterns = [
        "poll shows", "polling", "approval rating",
        "primary race", "primary election", "voters support",
        "favorability", "senate race", "congressional race"
    ]
    
    # Investigation/legal tangents
    investigation_patterns = [
        "investigation into", "indictment", "grand jury",
        "hearing", "testified", "criminal prosecution",
        "referred to justice", "doj charges"
    ]
    
    all_patterns = shutdown_patterns + electoral_patterns + investigation_patterns
    
    # Check if claim is primarily about these tangents
    matches = sum(1 for pattern in all_patterns if pattern in text_lower)
    
    # If 2+ tangent keywords, it's probably a tangent
    if matches >= 2:
        return True
    
    # Single strong indicator
    strong_tangents = ["government shutdown", "poll shows", "indictment", "senate blocked"]
    if any(pattern in text_lower for pattern in strong_tangents):
        return True
    
    return False


# ============================================================================
# CORE EXTRACTION: Factual Claims
# ============================================================================

def extract_factual_claims(text: str, nlp) -> list:

    doc = nlp(text)
    claims = []
    
    NAMED_LABELS = {"PERSON", "ORG", "GPE", "DATE", "LOC", "NORP", "FAC", "EVENT"}
    NUM_LABELS = {"CARDINAL", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "TIME"}
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        if len(sent_text) < 20 or sent_text.endswith('?'):
            continue
        
        # Extract entities for THIS sentence
        named_ents = [(ent.text, ent.label_) for ent in sent.ents 
                      if ent.label_ in NAMED_LABELS]
        
        numeric_ents = [(ent.text, ent.label_) for ent in sent.ents 
                        if ent.label_ in NUM_LABELS]
        
        # Pull out specific info
        dates = [e[0] for e in named_ents if e[1] == "DATE"]
        numbers = [e[0] for e in numeric_ents]
        
        # Determine claim type
        has_numbers = len(numeric_ents) > 0
        has_dates = len(dates) > 0
        has_person = any(e[1] == "PERSON" for e in named_ents)
        has_verb = any(token.pos_ == "VERB" for token in sent)
        
        reporting_verbs = {"say", "announce", "report", "confirm", "state", "reveal"}
        has_reporting = any(token.lemma_ in reporting_verbs for token in sent)
        
        # Prioritize claim type
        if has_reporting:
            claim_type = "reported_fact"
        elif has_numbers:
            claim_type = "numerical"
        elif has_dates:
            claim_type = "temporal"
        elif has_person and has_verb:
            claim_type = "person_action"
        else:
            claim_type = "general"
        
        claim = {
            "text": sent_text,
            "entities": named_ents,
            "numbers": numbers,
            "dates": dates,
            "claim_type": claim_type
        }
        
        claims.append(claim)
    
    return claims

# ============================================================================
# BATCH PROCESSING: Extract Claims from All Articles
# ============================================================================

def extract_all_claims_by_group(article_bank_path: str) -> dict:
    """Extract claims from all articles, organized by group"""
    
    # Load data
    with open(article_bank_path, 'r') as f:
        data = json.load(f)
    
    articles = data["articles"]
    groups = data["groups"]
    
    # Create fast lookup: article_id → article (eliminates nested loop inefficiency)
    article_by_id = {article["article_id"]: article for article in articles}
    
    nlp = load_spacy_model()
    claims_by_group = {}
    skipped = 0
    
    # Loop through each group
    for i, group in enumerate(groups):
        # Loop through article IDs in this group
        for article_id in group:
            article = article_by_id.get(article_id)
            
            if not article:
                continue
            
            text = article["full_text"]
            
            # Skip video pages
            if "/watch/" in article.get("url", ""):
                skipped += 1
                continue
            
            # Skip short articles
            if len(text) < 100:
                skipped += 1
                continue
            
            # Extract claims
            text = clean_text(text)
            claims = extract_factual_claims(text, nlp)
            
            # Store claims
            for claim in claims:
                if i not in claims_by_group:
                    claims_by_group[i] = []
                claims_by_group[i].append({
                    "claim": claim,
                    "source": article.get("source_domain", "Unknown"),
                    "article_title": article.get("title", ""),
                    "article_url": article.get("url", ""),
                    "article_id": article.get("article_id")
                })
    
    return claims_by_group

# ============================================================================
# Deciding which sentence represents the group
# ============================================================================
def select_representative(cluster: list, model, weights: dict, precomputed_embeddings=None):
    """
    Select the most representative claim from a cluster.
    
    Args:
        cluster: List of claim objects
        model: Embedding model (only used if precomputed_embeddings is None)
        weights: Sentiment lexicon weights
        precomputed_embeddings: Optional pre-computed embeddings (avoids re-encoding)
    """
    if len(cluster) == 1:
        return cluster[0]

    cluster_texts = [item["claim"]["text"] for item in cluster]
    cluster_texts_cleaned = [clean_text(t) for t in cluster_texts]

    # Use precomputed embeddings if available, otherwise compute them
    if precomputed_embeddings is not None:
        emb = precomputed_embeddings
    else:
        emb = model.encode_corpus(cluster_texts_cleaned)
    
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    centroid = emb_norm.mean(axis=0)
    c_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

    sim_scores = [float((np.dot(c_norm, emb_norm[i]) + 1.0) / 2.0) for i in range(len(cluster))]

    neutrality_scores = []
    for text_clean in cluster_texts_cleaned:
        loaded = rules.sentence_loaded(text_clean, weights)  # 0..1
        neutrality_scores.append(1.0 - loaded)

    completeness_scores = []
    for item in cluster:
        claim = item["claim"]
        score = 0.0
        if claim.get("numbers"): score += 0.25
        if claim.get("dates"):   score += 0.25
        ents = claim.get("entities", [])
        if any(lbl in {"GPE", "LOC"} for _, lbl in ents):    score += 0.25
        if any(lbl in {"PERSON", "ORG"} for _, lbl in ents): score += 0.25
        completeness_scores.append(min(1.0, score))

    clarity_scores = []
    for text in cluster_texts:
        t = text  # original
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", t)
        penalty = 0.0
        if len(tokens) < 6 or len(tokens) > 35:
            penalty += 0.20
        punct_hits = sum([t.count(","), t.count(";"), t.count("—"), t.count("–")])
        if punct_hits > 1:
            penalty += 0.20
        if re.search(r"\b(was|were|been|being|is|are|be)\s+\w+ed\b", t.lower()):
            penalty += 0.10
        clarity_scores.append(max(0.0, min(1.0, 1.0 - penalty)))

    final_scores = []
    for i in range(len(cluster)):
        final_i = (
            0.45 * sim_scores[i]
            + 0.30 * neutrality_scores[i]
            + 0.20 * completeness_scores[i]
            + 0.05 * clarity_scores[i]
        )
        final_scores.append(final_i)

    best = np.argmax(final_scores)
    top = final_scores[best]
    # just in case of a tie (can we be done already lol)
    tie_idxs = [i for i, s in enumerate(final_scores) if abs(s - top) <= 0.02]
    if len(tie_idxs) > 1:
        tie_idxs.sort(key=lambda i: (-neutrality_scores[i], -completeness_scores[i], len(cluster_texts[i])))
        best = tie_idxs[0]

    return cluster[best]

# ============================================================================
# Core Fact, Left Fact, Right Fact, and BS Claims
# ============================================================================
def load_outlet_lean(outlet_lean_path: str = None) -> dict:
    """Load the outlet lean scores from JSON file"""
    if outlet_lean_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        outlet_lean_path = os.path.join(current_dir, "..", "data", "outlet_lean.json")
    
    with open(outlet_lean_path, 'r') as f:
        return json.load(f)


def type_of_claim(cluster: list, outlet_lean: dict = None) -> str:
    """
    Classify a cluster of claims based on source distribution.
    
    More sophisticated logic that accounts for:
    - Source diversity (both left and right present)
    - Absolute numbers, not just percentages
    - Real-world bias in article availability
    
    Returns:
        'right_claim': Heavily right-skewed with no diversity
        'left_claim': Heavily left-skewed with no diversity
        'core_fact': Has cross-partisan coverage OR not heavily partisan
        'unverified': Only 2 claims, not enough validation
    """
    if outlet_lean is None:
        outlet_lean = load_outlet_lean()
    
    cluster_size = len(cluster)
    
    # Count left-leaning, right-leaning, and neutral sources
    left_count = 0
    right_count = 0
    neutral_count = 0
    
    for claim_obj in cluster:
        source = claim_obj.get("source", "Unknown")
        lean_score = outlet_lean.get(source, 0)
        
        if lean_score < 0:
            left_count += 1
        elif lean_score > 0:
            right_count += 1
        else:
            neutral_count += 1
    
    # Calculate percentages
    left_pct = left_count / cluster_size
    right_pct = right_count / cluster_size
    
    # KEY INSIGHT: If there's SOURCE DIVERSITY (both left and right present), 
    # it's likely a core fact regardless of the percentage split
    has_diversity = (left_count > 0 and right_count > 0)
    
    # More lenient thresholds for partisan claims
    # Only mark as partisan if it's HEAVILY skewed AND lacks diversity
    heavily_right = right_pct >= 0.80
    heavily_left = left_pct >= 0.80
    
    # Classification logic:
    # 1. If it has diversity (both left and right sources), it's a core fact
    if has_diversity and cluster_size >= 3:
        return "core_fact"
    
    # 2. If it's heavily skewed to one side with NO opposing voices, it's partisan
    if heavily_right and right_count >= 2 and left_count == 0:
        return "right_claim"
    elif heavily_left and left_count >= 2 and right_count == 0:
        return "left_claim"
    
    # 3. If it has 3+ sources but not heavily partisan, treat as core fact
    # (Even if it's 70% left, if there are some right sources or it's not extreme, it's valid)
    if cluster_size >= 3:
        return "core_fact"
    
    # 4. Only 2 sources - not enough for validation
    return "unverified"



# ===========================================================================
# Topic Coherence: Filter claims by relevance to main topic
# ============================================================================

def compute_main_topic_embedding(group_article_ids: list, article_lookup: dict, model) -> np.ndarray:
    """
    Compute the main topic embedding for a group based on article titles and descriptions.
    
    This represents what the group is ACTUALLY about.
    """
    topic_texts = []
    for article_id in group_article_ids:
        article = article_lookup.get(article_id)
        if article:
            # Use title + description (more focused than full text)
            title = article.get("title", "")
            desc = article.get("description", "")
            topic_texts.append(f"{title}. {desc}")
    
    if not topic_texts:
        return None
    
    # Encode and compute centroid
    embeddings = model.encode_corpus(topic_texts)
    centroid = np.mean(embeddings, axis=0)
    
    # Normalize
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    return centroid_norm

def is_relevant_to_topic(claim_embedding: np.ndarray, topic_embedding: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if a claim is relevant to the main topic.
    
    Args:
        claim_embedding: Normalized embedding of the claim
        topic_embedding: Normalized embedding of the main topic
        threshold: Minimum similarity (0.5 = somewhat related, 0.7 = very related)
    
    Returns:
        True if claim is relevant to the main topic
    """
    if topic_embedding is None:
        return True  # Can't filter without topic
    
    similarity = np.dot(claim_embedding, topic_embedding)
    return similarity >= threshold


# ===========================================================================
# Embedding Cache Management
# ============================================================================

def load_embedding_cache(cache_path: str) -> dict:
    """Load cached embeddings from numpy file"""
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True)
            embeddings_dict = {key: cache[key] for key in cache.files}
            print(f"✓ Loaded {len(embeddings_dict)} cached embeddings from {cache_path}")
            return embeddings_dict
        except Exception as e:
            print(f"⚠ Could not load cache: {e}. Starting fresh.")
            return {}
    return {}

def save_embedding_cache(cache_dict: dict, cache_path: str):
    """Save embeddings to numpy compressed file"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **cache_dict)
    print(f"✓ Saved {len(cache_dict)} embeddings to {cache_path}")

def generate_claim_id(group_id: str, claim_idx: int, claim_text: str) -> str:
    """Generate a unique ID for a claim based on content hash"""
    import hashlib
    # Use hash of text to ensure same claim always gets same ID
    text_hash = hashlib.md5(claim_text.encode()).hexdigest()[:8]
    return f"g{group_id}_c{claim_idx}_{text_hash}"


# ===========================================================================
# Clustering Claims by Group (with caching)
# ============================================================================
def cluster_claims_by_group(
    article_bank_path: str, 
    skip_extraction: bool = False,
    use_cache: bool = True,
    cache_path: str = None,
    topic_relevance_threshold: float = 0.60
) -> list:
    """
    Cluster claims by group with topic coherence filtering.
    
    Args:
        topic_relevance_threshold: Minimum similarity to main topic (0.60 = moderately strict)
                                   Lower = more inclusive, Higher = stricter
    """
    # Set up cache path
    if cache_path is None:
        data_dir = os.path.dirname(article_bank_path)
        cache_path = os.path.join(data_dir, "embeddings_cache.npz")
    
    # Load data (need it for topic modeling)
    with open(article_bank_path, 'r') as f:
        data = json.load(f)
    
    # Load or extract claims
    if skip_extraction:
        claims_by_group = data.get("claims_by_group")
        if not claims_by_group:
            raise ValueError("No claims_by_group found in file. Run with skip_extraction=False first.")
    else:
        # Extract claims and use return value (avoid re-reading from disk)
        claims_by_group = extract_all_claims_by_group(article_bank_path)
    
    # Build article lookup for topic modeling
    article_lookup = {a['article_id']: a for a in data['articles']}
    groups = data['groups']
    
    # Load embedding cache
    embeddings_cache = {}
    if use_cache:
        embeddings_cache = load_embedding_cache(cache_path)
    
    # Prepare to collect new embeddings
    model = None  # Load only if needed
    new_embeddings_count = 0
    filtered_count = 0
    
    # Result: list of groups, each containing list of claim clusters
    all_groups_clustered = []

    for group_id, group_claims in claims_by_group.items(): #group level
        # Get the article IDs for this group to compute topic embedding
        group_idx = int(group_id) if isinstance(group_id, str) else group_id
        if group_idx < len(groups):
            group_article_ids = groups[group_idx]
        else:
            group_article_ids = []
        
        # Filter claims by noise and background FIRST
        filtered_claims = []
        for claim_obj in group_claims:
            claim_text = claim_obj["claim"]["text"]
            claim_data = claim_obj["claim"]
            
            # Skip noise claims
            if is_noise_claim(claim_text):
                filtered_count += 1
                continue
            
            # Skip background/historical context
            if is_background_context(claim_text, claim_data):
                filtered_count += 1
                continue
            
            # Skip political tangents (shutdown, polls, investigations)
            if is_political_tangent(claim_text):
                filtered_count += 1
                continue
            
            filtered_claims.append(claim_obj)
        
        group_claims = filtered_claims
        claim_texts = [c["claim"]["text"] for c in group_claims]
        
        # Check which claims need new embeddings
        embeddings = []
        claims_to_embed = []
        claim_ids = []
        
        for idx, claim_obj in enumerate(group_claims):
            claim_text = claim_obj["claim"]["text"]
            claim_id = generate_claim_id(str(group_id), idx, claim_text)
            claim_ids.append(claim_id)
            
            if use_cache and claim_id in embeddings_cache:
                # Use cached embedding
                embeddings.append(embeddings_cache[claim_id])
            else:
                # Mark for new embedding computation
                claims_to_embed.append((idx, claim_text, claim_id))
                embeddings.append(None)  # Placeholder
        
        # Generate embeddings for new claims only
        if claims_to_embed:
            if model is None:
                model = load_embedding_model()
            
            texts_to_embed = [text for _, text, _ in claims_to_embed]
            new_embeddings = model.encode_corpus(texts_to_embed)
            
            # Insert new embeddings and update cache
            for i, (orig_idx, text, claim_id) in enumerate(claims_to_embed):
                embeddings[orig_idx] = new_embeddings[i]
                embeddings_cache[claim_id] = new_embeddings[i]
                new_embeddings_count += 1
            
            print(f"  Group {group_id}: Generated {len(claims_to_embed)} new embeddings, used {len(claim_texts) - len(claims_to_embed)} cached")
        else:
            print(f"  Group {group_id}: All {len(claim_texts)} embeddings from cache")
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Normalize embeddings for faster similarity computation
        # After normalization, dot product = cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # TOPIC COHERENCE FILTERING
        # Compute main topic embedding for this group
        topic_embedding = compute_main_topic_embedding(group_article_ids, article_lookup, model)
        
        # Filter claims by relevance to main topic
        relevant_indices = []
        for i in range(len(group_claims)):
            if is_relevant_to_topic(embeddings_norm[i], topic_embedding, topic_relevance_threshold):
                relevant_indices.append(i)
            else:
                filtered_count += 1
        
        # Keep only relevant claims and embeddings
        if len(relevant_indices) < len(group_claims):
            group_claims = [group_claims[i] for i in relevant_indices]
            embeddings_norm = embeddings_norm[relevant_indices]
            embeddings = embeddings[relevant_indices]
            print(f"  Group {group_id}: Filtered to {len(relevant_indices)}/{len(embeddings_norm) + len(relevant_indices)} relevant claims")
        
        # Recompute similarity matrix with filtered claims
        sim_matrix = embeddings_norm @ embeddings_norm.T  # Faster than cosine_similarity

        # 1. Build adjacency graph based on similarity threshold
        adj = {i: set() for i in range(len(group_claims))}
        for i in range(len(group_claims)):
            for j in range(i+1, len(group_claims)):
                if sim_matrix[i][j] >= 0.85:
                    adj[i].add(j)
                    adj[j].add(i)

        # 2. Find connected components (same as grouper)
        visited = set()
        clusters = []

        for i in range(len(group_claims)):
            if i in visited:
                continue
            
            # DFS to find all connected claims
            stack = [i]
            cluster = []
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            clusters.append(cluster)

        # 3. Convert indices to actual claim objects and store their embeddings
        clustered_claims = []
        cluster_embeddings = []
        for cluster in clusters:
            cluster_claims = [group_claims[idx] for idx in cluster]
            cluster_embs = embeddings[cluster]  # Get embeddings for this cluster
            clustered_claims.append(cluster_claims)
            cluster_embeddings.append(cluster_embs)

        all_groups_clustered.append({
            "clusters": clustered_claims,
            "embeddings": cluster_embeddings
        })
    
    # Save updated cache if we generated new embeddings
    if use_cache and new_embeddings_count > 0:
        save_embedding_cache(embeddings_cache, cache_path)
        print(f"\n✓ Total: {new_embeddings_count} new embeddings computed, {len(embeddings_cache) - new_embeddings_count} from cache")
    elif use_cache:
        print(f"\n✓ All {len(embeddings_cache)} embeddings loaded from cache (0 new)")
    
    # Show filtering results
    if filtered_count > 0:
        print(f"\n✓ Filtered out {filtered_count} off-topic/noise claims")
        print(f"  This improves factbank coherence and quality")
    
    return all_groups_clustered

def write_fact_bank(data_path: str, all_groups_clustered: list):
    """
    Build factbanks from clustered claims.
    
    Args:
        all_groups_clustered: List of dicts with 'clusters' and 'embeddings' keys
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    factbanks = []
    model = None  # Only load if needed (for backward compatibility)
    weights = load_sentiment_lexicon()
    
    for group_idx in range(len(all_groups_clustered)):
        group_data = all_groups_clustered[group_idx]
        
        # Handle both old format (list) and new format (dict with embeddings)
        if isinstance(group_data, dict):
            clusters = group_data["clusters"]
            embeddings = group_data["embeddings"]
        else:
            # Backward compatibility: old format without embeddings
            clusters = group_data
            embeddings = [None] * len(clusters)
            if model is None:
                model = load_embedding_model()
        
        factbank = {}
        core_facts = []
        left_claims = []
        right_claims = []
        
        for cluster_idx, cluster in enumerate(clusters):
            # Skip singleton claims (unclustered)
            if len(cluster) == 1:
                continue
            
            # For clustered claims (2+), classify by outlet lean
            claim_type = type_of_claim(cluster)
            
            # Use precomputed embeddings (NO re-encoding!)
            cluster_embs = embeddings[cluster_idx] if embeddings[cluster_idx] is not None else None
            representative_claim = select_representative(cluster, model, weights, cluster_embs)
            
            # Extract just the text and source
            claim_text = representative_claim["claim"]["text"]
            source = representative_claim.get("source", "Unknown")
            
            if claim_type == "core_fact":
                core_facts.append({
                    "id": f"F{len(core_facts) + 1}",
                    "text": claim_text
                })
            elif claim_type == "left_claim":
                left_claims.append({
                    "id": f"CL{len(left_claims) + 1}",
                    "text": claim_text,
                    "attribution": source
                })
            elif claim_type == "right_claim":
                right_claims.append({
                    "id": f"CR{len(right_claims) + 1}",
                    "text": claim_text,
                    "attribution": source
                })
            # Note: "unverified" claims (2 articles, not partisan) are skipped
        
        # Build factbank structure
        factbank["topic_id"] = f"topic_{group_idx}"
        factbank["core_facts"] = core_facts
        factbank["claims_left"] = left_claims
        factbank["claims_right"] = right_claims
        factbanks.append(factbank)

    data["factbanks"] = factbanks

    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Factbank written to {data_path}")
    return factbanks