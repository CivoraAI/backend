import json
import spacy
import re

def extract_quotes_with_attribution(text: str, nlp) -> list:
    """
    Extract quotes and their speakers using spaCy.
    Returns list of {"speaker": "...", "text": "..."}
    """
    quotes = []
    
    # Find all quoted text using regex
    quote_pattern = r'"([^"]+)"'
    matches = list(re.finditer(quote_pattern, text))
    
    # Process full text with spaCy
    doc = nlp(text)
    
    for match in matches:
        quote_text = match.group(1)
        quote_start = match.start()
        quote_end = match.end()
        
        # Get context around the quote (before and after)
        context_start = max(0, quote_start - 150)
        context_end = min(len(text), quote_end + 150)
        context = text[context_start:context_end]
        
        # Process context with spaCy
        context_doc = nlp(context)
        
        attribution = None
        
        # Look for attribution verbs (said, stated, told, etc.)
        attribution_verbs = {"say", "state", "announce", "tell", "claim", "argue", "note", "explain", "add", "warn"}
        
        for token in context_doc:
            if token.lemma_ in attribution_verbs:
                # Find the subject (who said it)
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        # Get full noun phrase
                        attribution = ' '.join([t.text for t in child.subtree])
                        break
                if attribution:
                    break
        
        # Fallback: Look for PERSON or ORG entities near the quote
        if not attribution:
            for ent in context_doc.ents:
                if ent.label_ in {"PERSON", "ORG"}:
                    # Check if entity is close to the quote
                    ent_start = ent.start_char + context_start
                    distance = min(abs(ent_start - quote_start), abs(ent_start - quote_end))
                    if distance < 100:  # Within 100 chars
                        attribution = ent.text
                        break
        
        if attribution:
            # Clean up attribution text
            attribution = ' '.join(attribution.split())  # Remove extra whitespace
            quotes.append({"speaker": attribution, "text": quote_text})
    
    return quotes

def sentences_quotes(data_path: str, process_all: bool = False) -> list:
    """
    Extract sentences and quotes from articles.
    
    Args:
        data_path: Path to news_data.json
        process_all: If True, process all articles. If False, only process new ones without 'sentences' field
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    articles = data["articles"]
    
    # Filter to only new articles if not processing all
    if not process_all:
        articles_to_process = [a for a in articles if 'sentences' not in a]
        if not articles_to_process:
            print("  No new articles to process")
            return data
    else:
        articles_to_process = articles
    
    print(f"  Processing {len(articles_to_process)} articles for quotes/sentences...")
    
    # Load spaCy model only if needed
    if articles_to_process:
        nlp = spacy.load("en_core_web_md")
    
    for i, article in enumerate(articles_to_process):
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(articles_to_process)} articles...")
        
        article["sentences"] = []
        article["quotes"] = []
        
        full_text = article.get("full_text", "")
        
        # Extract quotes first
        quotes = extract_quotes_with_attribution(full_text, nlp)
        article["quotes"] = quotes
        
        # Process text into sentences
        doc = nlp(full_text)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Check if sentence contains a quote
            has_quote = '"' in sent_text
            
            if has_quote:
                # This sentence has a quote - it's already captured in quotes array
                continue
            else:
                # Regular sentence - add to sentences
                if len(sent_text) > 20:  # Filter very short fragments
                    article["sentences"].append(sent_text)
    
    # Save updated data
    with open(data_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"  ✅ Processed {len(articles_to_process)} articles")
    return data

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/news_data.json"
    sentences_quotes(data_file)
