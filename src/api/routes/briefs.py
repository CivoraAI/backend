from fastapi import APIRouter
import json
import os

router = APIRouter()

DATA_PATH = "src/core/metrics/civai_bias/data/news_data.json"

@router.get("/briefs")
def get_briefs():
    """Return all briefs and their metadata from the factbank file"""
    if not os.path.exists(DATA_PATH):
        return {"briefs": [], "message": "No data file found"}
    
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    # extract just what we need for frontend display
    breifs_dicts = data.get("briefs", [])
    articles = data.get("articles", [])
    groups = data.get("groups", [])
    briefs = []
    
    for br in breifs_dicts:
        topic_id = br.get("topic_id", None)
        core_facts_brief = br.get("core_facts_brief", None)
        left_claims_brief = br.get("left_claims_brief", None)
        right_claims_brief = br.get("right_claims_brief", None)

        # Safety check: validate topic_id and groups
        if topic_id is None or not isinstance(topic_id, int):
            continue
        if topic_id >= len(groups) or topic_id < 0:
            continue
        
        # Get article IDs for this topic/group
        group_article_ids = set(groups[topic_id])
        
        # Efficiently extract all fields in one pass
        matching_articles = [
            article for article in articles 
            if article.get("article_id") in group_article_ids
        ]
        
        urls = [article.get("url", "") for article in matching_articles]
        titles = [article.get("title", "") for article in matching_articles]
        authors = [article.get("author") for article in matching_articles]  # Can be None
        published_dates = [article.get("publish_date", "") for article in matching_articles]

        fcs = [article.get("fc", "") for article in matching_articles]
        ocs = [article.get("oc", "") for article in matching_articles]
        sds = [article.get("sd", "") for article in matching_articles]
        lis = [article.get("li", "") for article in matching_articles]
        article_biases = [article.get("article_bias", "") for article in matching_articles]

        briefs.append({
            "topic_id": topic_id,
            "core_facts_brief": core_facts_brief,
            "left_claims_brief": left_claims_brief,
            "right_claims_brief": right_claims_brief,
            "urls": urls,
            "titles": titles,
            "authors": authors,
            "published_dates": published_dates,
            "fcs": fcs,
            "ocs": ocs,
            "sds": sds,
            "lis": lis,
            "article_biases": article_biases,
        })
    
    return {"briefs": briefs, "count": len(briefs)}

