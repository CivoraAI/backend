#!/opt/anaconda3/envs/civai_py310/bin/python
# metric functions: fact_coverage, opposition_coverage, source_diversity, loaded_intensity, sentiment

from ast import Break, Pass
import json
import os
import re
import signal
import math
import subprocess
import sys
from .text import normalize_speaker
from .rules import *


CORRECT_PYTHON = "/opt/anaconda3/envs/civai_py310/bin/python"

from sklearn.metrics.pairwise import cosine_similarity



########################################################
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
                cwd=os.path.dirname(os.path.dirname(__file__)),
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
        Pass


# Check and switch Python interpreter if needed
c################################################################################

# METHODS


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = re.sub(r"[.,;:!?“”‘’]", "", text)
    return text


def tokenize(text: str) -> set:
    tokens = text.split()
    stopwords = [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "by",
        "to",
        "for",
        "as",
        "was",
        "were",
        "is",
        "are",
        "within",
    ]
    tokens = [token for token in tokens if token not in stopwords]
    return set(tokens)

def normalize_lean(raw_val: float) -> float:
    """Normalize raw bias index (like -10..+10) to -1..+1 range."""
    # Cap absurd outliers just in case
    raw_val = max(-15.0, min(15.0, raw_val))
    return round(raw_val / 15.0, 3)  # now roughly in [-1, 1]

########################################################
#Metrics


def fact_coverage(file_path_article: str, file_path_factbank: str):
    check_and_switch_python()
    with open(file_path_article, "r") as f:
        article = json.load(f)
        sentences = article["sentences"]
        sentences_sets = [tokenize(sentence) for sentence in sentences]
    with open(file_path_factbank, "r") as f:
        data = json.load(f)
        core_facts = data["core_facts"]
        core_facts_texts = [fact["text"] for fact in core_facts]
        core_facts_sets = [tokenize(fact["text"]) for fact in core_facts]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU only
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from FlagEmbedding import FlagModel

    # Initialize model with minimal settings
    model = FlagModel(
        "BAAI/bge-base-en-v1.5", use_fp16=False, devices=["cpu"], normalize_embeddings=True
    )

    signal.alarm(0)  # Cancel timeout
    sentences_embeddings = model.encode_queries(sentences)
    core_facts_embeddings = model.encode_corpus(core_facts_texts)

    cosine_sim = cosine_similarity(sentences_embeddings, core_facts_embeddings)
    score = 0

    for core_fact_idx in range(len(core_facts_embeddings)):
        for sentence_idx in range(len(sentences_embeddings)):
            similarity = cosine_sim[sentence_idx][core_fact_idx]
            if similarity > 0.75:
                score += 1
                break
    fc = score / len(core_facts_sets)
    return fc

def source_diversity(file_path_article) -> float:
    check_and_switch_python()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU only
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    with open(file_path_article, "r") as f:
        article = json.load(f)
    alias_map={}
    uniq = set()
    for quote in article["quotes"]:
        raw_speaker=quote["speaker"]
        key=normalize_speaker(raw_speaker, alias_map)
        if key is not None and key != "":
            uniq.add(key)
    sd = min(1.0, len(uniq)/3)
    return sd



def opposition_coverage(file_path_article, file_path_factbank) -> float:
    check_and_switch_python()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU only
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    with open(file_path_article, "r") as f:
        article = json.load(f)
    article_domain = article["domain"]

    with open(file_path_factbank, "r") as f:
        factbank = json.load(f)

    with open("data/outlet_lean.json", "r") as f:
        outlet_lean = json.load(f)
    outlet_lean_score = outlet_lean[article_domain]
    outlet_lean_score = normalize_lean(outlet_lean_score)
    
    if outlet_lean_score > 0:
        opposing_claims = [c["text"] for c in factbank["claims_left"]]
    elif outlet_lean_score < 0:
        opposing_claims = [c["text"] for c in factbank["claims_right"]]
    
    from FlagEmbedding import FlagModel

    model = FlagModel(
        "BAAI/bge-base-en-v1.5", use_fp16=False, devices=["cpu"], normalize_embeddings=True
    )

    scores={}
    article_sentences_embeddings = model.encode_queries(article["sentences"])
    quote_texts = [quote["text"] for quote in article["quotes"]]
    article_quotes_embeddings = model.encode_queries(quote_texts)
    opposing_claims_embeddings = model.encode_corpus(opposing_claims)

    cosine_sim_sentences = cosine_similarity(article_sentences_embeddings, opposing_claims_embeddings)
    cosine_sim_quotes = cosine_similarity(article_quotes_embeddings, opposing_claims_embeddings)

    for i in range(len(cosine_sim_sentences)):
        for j in range(len(opposing_claims)):
            if cosine_sim_sentences[i][j] > 0.65:
                scores[j] = 0.5
                break
    for i in range(len(cosine_sim_quotes)):
        for j in range(len(opposing_claims)):
            if cosine_sim_quotes[i][j] > 0.65:
                if j not in scores:
                    scores[j] = 1
                    break
    
    oc=0
    for i in range(len(scores)):
        oc += scores[i]
    
    if len(opposing_claims) == 0:
        return 0.0, outlet_lean_score 
    oc=oc/len(opposing_claims)
    return oc, outlet_lean_score

def loaded_intensity(file_path_article: str, weights: dict, trim: float = 0.10) -> float:
    check_and_switch_python()
    with open(file_path_article, "r") as f:
        article = json.load(f)
    sentences = article["sentences"]
    if not sentences:
        return 0.0
    scores = [sentence_loaded(s, weights) for s in sentences]
    if len(scores) >=10:
        xs = sorted(scores)
        k = math.ceil(len(xs) * trim)
        mid = xs[k: len(xs)-k] or xs
        return sum(mid)/len(mid)
    else:
        return sum(scores)/len(scores)
    
########################################################
#BIG BOY

def article_bias(file_path_article: str, file_path_factbank: str, weights: dict, trim: float = 0.10) -> float:
    check_and_switch_python()
    fc = fact_coverage(file_path_article, file_path_factbank)
    oc, L = opposition_coverage(file_path_article, file_path_factbank)
    sd = source_diversity(file_path_article)
    li = loaded_intensity(file_path_article, weights, trim)
    if L<0:
        lean_sign=-1
    else:
        lean_sign=1
    omission = 1 - (0.6 * fc + 0.3 * oc + 0.1 * sd)
    B = 0.9 * omission + 0.1 * li
    article_bias = max(-1.0, min(1.0, lean_sign * B))
    return article_bias

def load_unified_articles():
    """Load articles from the unified articles.json file"""
    import os
    articles_path = "data/articles/articles.json"
    
    if not os.path.exists(articles_path):
        # Fallback to individual files if unified doesn't exist
        return {
            "A_left": "data/articles/trial_topic/A_left.json",
            "B_right": "data/articles/trial_topic/B_right.json", 
            "C_center": "data/articles/trial_topic/C_center.json",
        }
    
    # Create temporary individual files from unified structure
    with open(articles_path, 'r') as f:
        unified_data = json.load(f)
    
    temp_articles = {}
    for article in unified_data["articles"]:
        article_id = article["id"]
        temp_path = f"temp_{article_id}.json"
        
        # Write individual article file temporarily
        with open(temp_path, 'w') as af:
            json.dump(article, af, indent=2)
        
        temp_articles[article_id] = temp_path
    
    return temp_articles

def cleanup_temp_files(temp_articles):
    """Clean up temporary article files"""
    import os
    for temp_path in temp_articles.values():
        if os.path.exists(temp_path) and temp_path.startswith("temp_"):
            os.remove(temp_path)

if __name__ == "__main__":
    weights = load_sentiment_lexicon()
    
    # Load articles from unified structure
    test_articles = load_unified_articles()
    factbank = "data/factbanks/trial_topic.json"

    try:
        for name, path in test_articles.items():
            print(f"\n=== {name.upper()} ===")
            fc = fact_coverage(path, factbank)
            oc, L = opposition_coverage(path, factbank)
            sd = source_diversity(path)
            li = loaded_intensity(path, weights)

            if L<0:
                lean_sign=-1
            else:
                lean_sign=1
            omission = 1 - (0.6 * fc + 0.3 * oc + 0.1 * sd)
            B = 0.9 * omission + 0.1 * li
            article_bias = max(-1.0, min(1.0, lean_sign * B))

            print(f"Fact Coverage (fc):       {fc:.3f}")
            print(f"Opposition Coverage (oc): {oc:.3f}")
            print(f"Source Diversity (sd):    {sd:.3f}")
            print(f"Loaded Intensity (li):    {li:.3f}")
            print(f"Omission score:           {omission:.3f}")
            print(f"Bias component (B):       {B:.3f}")
            print(f"Outlet lean (L):          {L:.3f}")
            print(f"→ Final Article Bias:     {article_bias:.3f}")
    finally:
        # Clean up temporary files
        cleanup_temp_files(test_articles)