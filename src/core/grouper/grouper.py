# article_clustering.py
import sys
import os
from openai import OpenAI
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def cluster_articles(articles_file_path: str, system_prompt_path: str, threshold: float = 0.7) -> List[List[int]]:
    """
    Cluster articles based on semantic similarity using OpenAI embeddings.
    
    Args:
        articles_file_path: Path to the articles file (separated by ⸻)
        system_prompt_path: Path to the system prompt file
        threshold: Similarity threshold for grouping articles (default: 0.7)
    
    Returns:
        List of groups, where each group is a list of article indices
    """
    # Set up API key - try multiple methods
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key)
    MODEL = "gpt-4o" 

    # Read system prompt
    with open(system_prompt_path, 'r') as file:
        content = file.read()

    # Read articles
    with open(articles_file_path, 'r') as file:
        articles_text = file.read()

    articles = articles_text.split("⸻")

    def extract(article):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": article},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return resp.choices[0].message.content

    # Extract content from articles
    articles_extracted = []
    for article in articles:
        articles_extracted.append(extract(article))

    # Generate embeddings
    embeddings = []
    for text in articles_extracted:
        resp = client.embeddings.create(
            model="text-embedding-3-small",  # or "text-embedding-3-large" for higher accuracy
            input=text
        )
        embeddings.append(resp.data[0].embedding)

    embeddings = np.array(embeddings)

    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Build adjacency list
    n = len(articles)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i][j] >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # DFS to find connected groups
    visited = set()
    groups = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nei in adj[node]:
                if nei not in visited:
                    stack.append(nei)
        groups.append(comp)

    return groups

def print_groups(groups: List[List[int]]) -> None:
    """Print the clustered groups in a readable format."""
    print("\n=== Threshold-based groups ===")
    for gidx, g in enumerate(groups):
        print(f"Group {gidx}:")
        for idx in g:
            print(f"  - Article {idx}")

if __name__ == "__main__":
    # Example usage
    articles_file = '/Users/arav/Documents/Civora/Scripts/article_sim_trials/iter_4/articles.txt'
    system_prompt_file = '/Users/arav/Documents/Civora/Scripts/article_sim_trials/iter_2/system_prompt.txt'
    
    groups = cluster_articles(articles_file, system_prompt_file, threshold=0.7)
    print_groups(groups)

