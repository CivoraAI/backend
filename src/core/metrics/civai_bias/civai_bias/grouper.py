# article_clustering.py
import os
from typing import List, Tuple
import numpy as np
import sys
import subprocess
import json
import re
from sklearn.metrics.pairwise import cosine_similarity


CORRECT_PYTHON = "/opt/anaconda3/envs/civai_py310/bin/python"

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
        pooling_method="cls"
    )

def load_articles(article_bank_path):
    with open(article_bank_path, 'r') as file:
        data = json.load(file)
    
    # Handle both old format (just list) and new format (dict with "articles" key)
    if isinstance(data, list):
        return data
    else:
        return data.get("articles", [])

def cluster_articles(articles_file_path: str, threshold: float = 0.7, allow_multi_group: bool = True) -> Tuple[List[List[int]], np.ndarray]:
    embedding_model = load_embedding_model()

    # Read articles
    articles = load_articles(articles_file_path)
    
    article_texts = []
    for article in articles:
        title = article["title"]
        description = article["description"]
        full_text = article["full_text"]
        
        # Combine them with spaces
        combined_text = f"{title} {description} {full_text}"
        article_texts.append(combined_text)

    # Generate embeddings using FlagEmbedding
    embeddings = embedding_model.encode_corpus(article_texts)
    embeddings = np.array(embeddings)

    sim_matrix = cosine_similarity(embeddings)

    # Build adjacency list
    n = len(articles)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i][j] >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Find groups
    if allow_multi_group:
        # Multi-group: articles can join multiple groups if they have high similarity
        groups = []
        
        # For each article, create a group with all its similar articles
        for i in range(n):
            if len(adj[i]) > 0:  # Only if article has connections
                group = [i] + list(adj[i])
                group.sort()  # Sort for consistent ordering
                
                # Only add if this exact group doesn't already exist
                if group not in groups:
                    groups.append(group)
        
        # Remove groups with less than 5 articles and sort by size
        groups = [g for g in groups if len(g) >= 4]
        groups.sort(key=len, reverse=True)
        
        # Remove groups that are subsets or have high overlap with larger groups
        filtered_groups = []
        for i, group in enumerate(groups):
            should_keep = True
            for j, other_group in enumerate(groups):
                if i != j:
                    # Check if current group is subset of other group
                    if set(group).issubset(set(other_group)):
                        should_keep = False
                        break
                    
                    # Check if groups have high overlap (>= 70% shared articles)
                    if len(group) > 1 and len(other_group) > 1:
                        intersection = set(group) & set(other_group)
                        overlap_ratio = len(intersection) / min(len(group), len(other_group))
                        
                        # If high overlap, prefer the group that comes first (larger groups are sorted first)
                        if overlap_ratio >= 0.75:
                            if len(other_group) > len(group):
                                should_keep = False
                                break
                            elif len(other_group) == len(group) and j < i:
                                # Same size groups: keep the one that appears first in the sorted list
                                should_keep = False
                                break
            
            if should_keep:
                filtered_groups.append(group)
        groups = filtered_groups
        
    else:
        # Original single-group approach: connected components
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
        
        # Filter out groups with less than 5 articles
        groups = [g for g in groups if len(g) >= 4]

    return groups, sim_matrix

def print_groups(groups: List[List[int]]) -> None:
    """Print the clustered groups in a readable format."""
    print("\n=== Groups ===")
    for gidx, g in enumerate(groups):
        print(f"Group {gidx}: {g}")

def print_similarity_matrix(sim_matrix: np.ndarray, threshold: float = 0.7) -> None:
    """Print similarity matrix for debugging."""
    print(f"\n=== Similarity Matrix (threshold: {threshold}) ===")
    n = sim_matrix.shape[0]
    print("     ", end="")
    for j in range(n):
        print(f"{j:6}", end="")
    print()
    
    for i in range(n):
        print(f"{i:3}: ", end="")
        for j in range(n):
            score = sim_matrix[i][j]
            if score >= threshold:
                print(f"{score:.3f}*", end=" ")
            else:
                print(f"{score:.3f} ", end=" ")
        print()
    print("* indicates similarity above threshold")

def assign_article_ids(articles_file_path: str):
    """
    One-time function to assign sequential IDs to existing articles.
    Only assigns IDs to articles that don't already have one.
    """
    # Load full data structure
    with open(articles_file_path, 'r') as file:
        data = json.load(file)
    
    # Handle both old format (just list) and new format (dict with "articles" key)
    if isinstance(data, list):
        articles = data
        data = {"articles": articles, "groups": {}}
    else:
        articles = data.get("articles", [])
    
    # Find the highest existing ID (if any)
    max_id = 0
    for article in articles:
        if 'article_id' in article and article['article_id'] is not None:
            max_id = max(max_id, article['article_id'])
    
    # Assign IDs to articles that don't have one
    next_id = max_id + 1
    assigned_count = 0
    for article in articles:
        if 'article_id' not in article or article['article_id'] is None:
            article['article_id'] = next_id
            next_id += 1
            assigned_count += 1
    
    # Update data structure
    data["articles"] = articles
    
    # Save back to file with full structure
    with open(articles_file_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Assigned IDs to {assigned_count} articles")
    print(f"✅ Total articles: {len(articles)}")
    print(f"✅ ID range: 1 to {next_id - 1}")
    
    return articles

def add_groups_to_articles(articles_file_path: str, threshold: float = 0.7, allow_multi_group: bool = True):
    """
    Read articles, cluster them, create 'groups' dict mapping group IDs to article IDs, save back to same file.
    """
    
    # Step 1: Load the full data structure
    with open(articles_file_path, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (just list) and new format (dict with "articles" key)
    if isinstance(data, list):
        articles = data
        data = {"articles": articles}
    else:
        articles = data.get("articles", [])
    
    # Step 1.5: Ensure all articles have IDs (assign if missing)
    max_id = 0
    for article in articles:
        if 'article_id' in article and article['article_id'] is not None:
            max_id = max(max_id, article['article_id'])
    
    next_id = max_id + 1
    for article in articles:
        if 'article_id' not in article or article['article_id'] is None:
            article['article_id'] = next_id
            next_id += 1
    
    # Step 2: Run clustering (reuse existing logic)
    groups, sim_matrix = cluster_articles(articles_file_path, threshold, allow_multi_group)
    
    # Step 3: Remove old 'group' property from articles if it exists
    for article in articles:
        if 'group' in article:
            del article['group']
    
    # Step 4: Create groups array where index = group_id
    # This ensures group IDs are true integers (not string keys)
    groups_array = []
    for group_id, article_indices in enumerate(groups):
        # Get the actual article IDs (not just array indices)
        article_ids = [articles[idx]['article_id'] for idx in article_indices]
        # Sort article IDs numerically within each group
        article_ids.sort()
        groups_array.append(article_ids)
    
    # Step 5: Update data structure with groups
    data["articles"] = articles
    data["groups"] = groups_array  # Array where index is the group ID
    
    # Step 6: Save back to JSON
    with open(articles_file_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processed {len(articles)} articles")
    print(f"✅ Created {len(groups)} groups")
    print(f"✅ Updated {articles_file_path}")

    return articles, groups, sim_matrix

if __name__ == "__main__":
    # Configuration
    articles_file = '/Users/arav/Documents/GitHub/backend/src/core/metrics/civai_bias/data/news_data.json'
    threshold = 0.80  # Threshold for similarity (tightened for more focused groups)
    
    # Step 1: Assign IDs to any articles that don't have them (one-time for existing articles)
    print("=== Assigning Article IDs ===")
    assign_article_ids(articles_file)
    
    # Step 2: Run the grouping and update the file
    print("\n=== Running Article Clustering ===")
    # Set allow_multi_group=True to allow articles to join multiple groups
    # Set allow_multi_group=False for traditional single-group clustering
    articles, groups, sim_matrix = add_groups_to_articles(
        articles_file, 
        threshold=threshold, 
        allow_multi_group=True  # Change this to False for single-group mode
    )
    
    # Step 3: Print group info
    print_groups(groups)
