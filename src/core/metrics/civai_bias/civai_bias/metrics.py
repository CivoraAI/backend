# metric functions: fact_coverage, opposition_coverage, source_diversity, loaded_intensity, sentiment

import json
import os
import re
import signal
import subprocess
import sys

from sklearn.metrics.pairwise import cosine_similarity

CORRECT_PYTHON = "/opt/anaconda3/envs/civai_py310/bin/python"


########################################################
def check_and_switch_python():
    """Check if we're using the correct Python interpreter and switch if needed"""
    print("=== PYTHON INTERPRETER CHECK ===")
    print(f"Current Python: {sys.executable}")
    print(f"Target Python: {CORRECT_PYTHON}")

    # Check if we're already using the correct Python
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
            print("Embedding script completed successfully!")
            sys.exit(0)  # Exit the current process since we've delegated to the correct one
        except subprocess.CalledProcessError as e:
            print(f"Error running with correct Python: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Python interpreter not found: {CORRECT_PYTHON}")
            print("Please check your Anaconda installation")
            print("Continuing with current Python interpreter...")
    else:
        print("Using correct Python interpreter!")


# Check and switch Python interpreter if needed
check_and_switch_python()
################################################################################

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

    # try:
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
            print(
                f"Sentence:{sentences_sets[sentence_idx]}, Core Fact:{core_facts_sets[core_fact_idx]}, Similarity:{similarity}"
            )
            if similarity > 0.75:
                score += 1
            for i in range(5):
                print()
    fc = score / len(core_facts_sets)
    print(fc)


fact_coverage("data/articles/trial_topic/B_right.json", "data/factbanks/trial_topic.json")
