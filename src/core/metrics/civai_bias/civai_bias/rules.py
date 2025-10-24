# helpers for outlet side detection and speaker-side cues

import csv
import re
import math 
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

_VADER = None

SENTIMENT_LEXICON_PATH = "data/lexicons/loaded_vader_terms.csv"

NEGATE =\
    ("aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite")

ATTENUATORS_SINGLE = \
    ("almost", "barely", "hardly", "just enough",
     "kind of", "kinda", "kindof", "kind-of",
     "less", "little", "marginal", "marginally",
     "occasional", "occasionally", "partly",
     "scarce", "scarcely", "slight", "slightly", "somewhat",
     "sort of", "sorta", "sortof", "sort-of")

ATTENUATORS_MULTI = \
    ["kind of",
     "sort of",
     "just enough"]

INTENSIFIERS_SINGLE = \
    ("absolutely", "amazingly", "awfully",
     "completely", "considerable", "considerably",
     "decidedly", "deeply", "effing", "enormous", "enormously",
     "entirely", "especially", "exceptional", "exceptionally",
     "extreme", "extremely",
     "fabulously", "flipping", "flippin", "frackin", "fracking",
     "fricking", "frickin", "frigging", "friggin", "fully",
     "fuckin", "fucking", "fuggin", "fugging",
     "greatly", "hella", "highly", "hugely",
     "incredible", "incredibly", "intensely",
     "major", "majorly", "more", "most", "particularly",
     "purely", "quite", "really", "remarkably",
     "so", "substantially",
     "thoroughly", "total", "totally", "tremendous", "tremendously",
     "uber", "unbelievably", "unusually", "utter", "utterly",
     "very")

    

def load_sentiment_lexicon():
    weights = {}
    with open(SENTIMENT_LEXICON_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            token = row[0].lower().strip()
            mean = float(row[1])
            std = float(row[2])
            conf = 1 / (1 + std)
            raw = abs(mean) * conf
            weight = min(3.0, 3.0 * raw / 4.0)
            weights[token] = weight
    return weights

def score_sentence(sentence: str, weights: dict) -> float:
    s = sentence.lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    tokens = [t for t in s.split() if t]

    counts = {} 

    for tok in tokens:
        base = None
        if tok in weights:
            base = tok
        else:
            # length guards before slicing
            if len(tok) >= 3 and tok.endswith("es") and tok[:-2] in weights:
                base = tok[:-2]
            elif len(tok) >= 2 and tok.endswith("s") and tok[:-1] in weights:
                base = tok[:-1]

        if base is not None:
            counts[base] = counts.get(base, 0) + 1

    # cap repeats at 2 and sum weights
    raw_sum = 0.0
    for base, ct in counts.items():
        raw_sum += min(ct, 2) * float(weights[base])

    return raw_sum

def raw_to_score(raw: float, s: float = 2.5) -> float:
    return 1 - math.exp(-raw / s)

def sentence_modifiers(sentence: str) -> tuple[bool, bool, bool]:
    # Step 1: Normalize and tokenize once
    s = sentence.lower()
    toks = re.sub(r"[^\w\s']", " ", s).split()
    
    # Step 2: Initialize flags once (never set back to False)
    has_neg = has_intensifier = has_attenuator = False
    
    # Step 3: Phrase pass first (multi-word attenuators)
    sp = re.sub(r"\s+", " ", s).strip()
    for phrase in ATTENUATORS_MULTI:
        if f" {phrase} " in f" {sp} ":
            has_attenuator = True
    
    # Step 4: Token pass (single words)
    for tok in toks:
        if tok in NEGATE:
            has_neg = True
        if tok in INTENSIFIERS_SINGLE:
            has_intensifier = True
        if tok in ATTENUATORS_SINGLE:
            has_attenuator = True
    
    # Step 5: Return the tuple
    return (has_neg, has_intensifier, has_attenuator)

def word_score(sentence: str, weights: dict) -> float:
    raw = score_sentence(sentence, weights)
    (has_neg, has_intens, has_atten) = sentence_modifiers(sentence)
    if has_neg:
        raw *= 0.9
    if has_intens:
        raw *= 1.15
    if has_atten:
        raw *= 0.9
    
    raw = min(raw, 10.0)
    return raw_to_score(raw)

def get_vader():
    global _VADER
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return _VADER

def vader_abs(sentence: str) -> float:
    an = get_vader()
    c = an.polarity_scores(sentence)["compound"]
    return min(1.0, max(0.0, abs(c)))

def sentence_loaded(sentence: str, weights: dict, alpha: float = 0.7) -> float:
    ws = word_score(sentence, weights)
    t  = vader_abs(sentence)     
    beta = 1 - alpha
    val = alpha*ws + beta*t
    return min(1.0, max(0.0, val))

weights = load_sentiment_lexicon()
print(sentence_loaded("This was an amazing, catastrophic failure.", weights))
print(sentence_loaded("It was a neutral statement.", weights))
print(sentence_loaded("That was not terrible.", weights))
