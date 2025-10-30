#!/usr/bin/env python3
import os
import json
import math
from typing import List, Dict

from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from civai_bias.rules import sentence_loaded, load_sentiment_lexicon


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "news_data.json")
OUTLET_LEAN_PATH = os.path.join(os.path.dirname(__file__), "data", "outlet_lean.json")


def tokenize(text: str) -> set:
    text = (text or "").lower().strip()
    stopwords = {
        "a", "an", "the", "of", "in", "on", "by", "to", "for", "as", "was", "were", "is", "are", "within",
    }
    return set([t for t in text.split() if t and t not in stopwords])


def compute_fact_coverage(sentences: List[str], core_facts: List[Dict]) -> float:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from FlagEmbedding import FlagModel

    core_facts_texts = [f.get("text", "") for f in core_facts]
    if not core_facts_texts:
        return 0.0

    model = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=False, devices=["cpu"], normalize_embeddings=True)
    sentences_embeddings = model.encode_queries(sentences or [])
    core_facts_embeddings = model.encode_corpus(core_facts_texts)

    cos = cosine_similarity(sentences_embeddings, core_facts_embeddings) if len(sentences_embeddings) and len(core_facts_embeddings) else []
    covered = 0
    for j in range(len(core_facts_embeddings)):
        found = False
        for i in range(len(sentences_embeddings)):
            if cos[i][j] > 0.75:
                found = True
                break
        if found:
            covered += 1
    return covered / len(core_facts_embeddings) if len(core_facts_embeddings) else 0.0


def normalize_lean(raw_val: float) -> float:
    raw_val = max(-15.0, min(15.0, float(raw_val)))
    return round(raw_val / 15.0, 3)


def compute_opposition_coverage(article: Dict, factbank: Dict, outlet_lean: Dict) -> (float, float):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from FlagEmbedding import FlagModel

    domain = article.get("source_domain", "")
    L = normalize_lean(outlet_lean.get(domain, 0))

    if L > 0:
        opposing_claims = [c.get("text", "") for c in factbank.get("claims_left", [])]
    elif L < 0:
        opposing_claims = [c.get("text", "") for c in factbank.get("claims_right", [])]
    else:
        opposing_claims = []

    model = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=False, devices=["cpu"], normalize_embeddings=True)
    sentence_texts = article.get("sentences", [])
    quote_texts = [q.get("text", "") for q in article.get("quotes", [])]

    if not opposing_claims:
        return 0.0, L

    emb_sent = model.encode_queries(sentence_texts) if sentence_texts else []
    emb_quotes = model.encode_queries(quote_texts) if quote_texts else []
    emb_opp = model.encode_corpus(opposing_claims) if opposing_claims else []

    sim_s = cosine_similarity(emb_sent, emb_opp) if (len(emb_sent) and len(emb_opp)) else []
    sim_q = cosine_similarity(emb_quotes, emb_opp) if (len(emb_quotes) and len(emb_opp)) else []

    scores = {}
    for i in range(len(sim_s)):
        for j in range(len(opposing_claims)):
            if sim_s[i][j] > 0.65:
                scores[j] = max(scores.get(j, 0), 0.5)
                break
    for i in range(len(sim_q)):
        for j in range(len(opposing_claims)):
            if sim_q[i][j] > 0.65:
                scores[j] = max(scores.get(j, 0), 1.0)
                break

    oc = sum(scores.values()) / len(opposing_claims) if opposing_claims else 0.0
    return oc, L


def compute_source_diversity(article: Dict) -> float:
    uniq = set()
    for q in article.get("quotes", []):
        sp = (q.get("speaker") or "").strip().lower()
        if sp:
            uniq.add(sp)
    return min(1.0, len(uniq) / 3.0)


def compute_loaded_intensity(article: Dict, weights: dict, trim: float = 0.10) -> float:
    sentences = article.get("sentences", [])
    if not sentences:
        return 0.0
    scores = [sentence_loaded(s, weights) for s in sentences]
    if len(scores) >= 10:
        xs = sorted(scores)
        k = math.ceil(len(xs) * trim)
        mid = xs[k: len(xs)-k] or xs
        return sum(mid) / len(mid)
    return sum(scores) / len(scores)


def compute_all_biases():
    with open(DATA_PATH) as f:
        data = json.load(f)

    articles = data.get("articles", [])
    groups = data.get("groups", [])
    factbanks = data.get("factbanks", [])

    with open(OUTLET_LEAN_PATH) as f:
        outlet_lean = json.load(f)

    weights = load_sentiment_lexicon()

    topic_to_factbank = {fb.get("topic_id"): fb for fb in factbanks}

    article_id_to_article = {a.get("article_id"): a for a in articles}

    for topic_id, group_article_ids in enumerate(groups):
        fb = topic_to_factbank.get(topic_id, {})
        core_facts = fb.get("core_facts", [])
        for aid in group_article_ids:
            a = article_id_to_article.get(aid)
            if not a:
                continue

            fc = compute_fact_coverage(a.get("sentences", []), core_facts)
            oc, L = compute_opposition_coverage(a, fb, outlet_lean)
            sd = compute_source_diversity(a)
            li = compute_loaded_intensity(a, weights)

            lean_sign = -1 if L < 0 else 1
            omission = 1 - (0.6 * fc + 0.3 * oc + 0.1 * sd)
            B = 0.9 * omission + 0.1 * li
            bias = max(-1.0, min(1.0, lean_sign * B))

            a["fc"] = round(fc, 6)
            a["oc"] = round(oc, 6)
            a["sd"] = round(sd, 6)
            a["li"] = round(li, 6)
            a["article_bias"] = round(bias, 6)

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    compute_all_biases()


