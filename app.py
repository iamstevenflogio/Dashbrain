import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
MODEL_NAME = "all-MiniLM-L6-v2"
JSON_PATH = "issue_cards.json"
TOP_K = 3
MIN_SCORE = 0.40

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity using pure Numpy (no sklearn needed)."""
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-8)

app = Flask(__name__)

# --- Data Processing Functions ---

def build_search_text(card: Dict[str, Any]) -> str:
    """Combines relevant fields from a card into a single string for embedding."""
    actions = " ".join(card.get("actions", []))
    tags = ", ".join(card.get("tags", []))
    recommended_fix = " ".join(card.get("recommended_fix", []))
    
    parts = [
        f"Ticket: {card.get('ticket_id', '')}",
        f"Lab: {card.get('lab', '')}",
        f"Module: {card.get('module', '')}",
        f"Status: {card.get('status', '')}",
        f"Concern: {card.get('concern_summary', '')}",
        f"Root cause: {card.get('root_cause', '')}",
        f"Recommended fix: {recommended_fix}",
        f"Actions: {actions}",
        f"Solver: {card.get('solver', '')}",
        f"Tags: {tags}",
    ]
    return "\n".join(parts)


def load_cards(path: str) -> List[Dict[str, Any]]:
    """Safely loads and filters the JSON database."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Starting with an empty database.")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Ensure we only process items that actually have a ticket ID
        return [c for c in data if c.get('ticket_id')]
    except json.JSONDecodeError:
        print(f"Error: {path} contains invalid JSON. Please check the file syntax.")
        return []


def search_cards(query: str, cards: List[Dict], embeddings: np.ndarray) -> List[Dict[str, Any]]:
    """Performs semantic search and applies MIN_SCORE threshold BEFORE applying TOP_K."""
    if embeddings.size == 0:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    # 1. Find all indices that meet the minimum score threshold
    valid_indices = np.where(scores >= MIN_SCORE)[0]
    
    if valid_indices.size == 0:
        return []

    # 2. Sort the valid indices by their score (descending)
    sorted_valid_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    
    # 3. Grab only the TOP_K from the valid results
    top_indices = sorted_valid_indices[:TOP_K]

    return [{'card': cards[idx], 'score': float(scores[idx])} for idx in top_indices]


# --- App Initialization ---
# NOTE: Loading the model and computing embeddings at the module level is great 
# for local development. If you deploy to production using a WSGI server like 
# Gunicorn with multiple workers, you MUST use the `--preload` flag, otherwise 
# every worker will download/load the model into RAM, causing memory crashes.

print("Loading ML model and computing embeddings... (This may take a moment)")
model = SentenceTransformer(MODEL_NAME)
cards = load_cards(JSON_PATH)
card_embeddings = np.load('card_embeddings.npy')
print(f"Initialization complete. Loaded {len(cards)} issue cards.")


# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    results = None
    best_card = None
    best_score = None
    no_match_message = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            results = search_cards(query, cards, card_embeddings)
            
            if results:
                best_card = results[0]['card']
                best_score = results[0]['score']
            else:
                no_match_message = (
                    "No sufficient match was found. "
                    "Please add more system-related tickets to the database to improve search results."
                )

    return render_template(
        'index.html',
        query=query,
        results=results,
        best_card=best_card,
        best_score=best_score,
        no_match_message=no_match_message
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)