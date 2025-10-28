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
    factbanks = data.get("briefs", [])
    briefs = []
    for fb in factbanks:
        topic_id = fb.get("topic_id", None)
        brief_text = fb.get("brief_text", None)
        briefs.append({
            "topic_id": topic_id,
            "brief_text": brief_text
        })
    
    return {"briefs": briefs, "count": len(briefs)}
    