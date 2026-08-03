import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import re

# --- Configuration ---
MODEL_NAME = "all-MiniLM-L6-v2"
JSON_PATH = "issue_cards.json"
TOP_K = 5
MIN_SCORE = 0.50

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

def extract_concern(query_text: str) -> str:
    """Looks for 'Concern:' in the pasted ticket and grabs everything after it.
    Falls back to the raw query text if the label isn't found."""
    match = re.search(r'Concern:\s*(.*)', query_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return query_text.strip()


def extract_ticket_id(query_text: str) -> Optional[str]:
    """Looks for a ticket ID pattern (e.g. DASH-14290, CS-105) anywhere in the query."""
    match = re.search(r'\b([A-Z]{2,6}-\d{2,6})\b', query_text.upper())
    return match.group(1) if match else None


def singularize(word: str) -> str:
    """Very light English de-pluralization, just enough to stop 'signatory' vs
    'signatories' from being treated as unrelated tokens by the keyword match."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        stem = word[:-2]
        # "-es" is a true plural suffix only after a sibilant ending
        # (box/boxes, dish/dishes, watch/watches, buzz/buzzes, index/indexes).
        # Everything else already ends in "e" in its singular form
        # (service/services, file/files, template/templates) — just drop the "s".
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def normalize_text(text: str) -> str:
    """Lowercases and singularizes each word, keeping word order/phrasing intact
    so multi-word tags/module names still need to match as a phrase, not just
    any overlapping word."""
    words = re.findall(r'[a-z0-9\-]+', text.lower())
    return " ".join(singularize(w) for w in words)


def keyword_match_count(query: str, card: Dict[str, Any]) -> int:
    """Counts how many of the card's module/tags appear as a phrase in the
    query text (plural-insensitive). A card matching several specific tags
    should rank above one that only shares one generic tag with many other
    cards (e.g. a broad 'signatories' tag shared across delete/add/visibility
    concerns alike)."""
    q_norm = normalize_text(query)
    count = 0
    module = card.get("module", "")
    if isinstance(module, str) and normalize_text(module) in q_norm:
        count += 1
    for tag in card.get("tags", []):
        if isinstance(tag, str) and normalize_text(tag) in q_norm:
            count += 1
    return count


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
    """Performs semantic search, then applies keyword boosts + quick intent penalty."""
    if embeddings.size == 0:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    q_norm = normalize_text(query)
    is_add_product_query = "add product" in q_norm and "price" not in q_norm

    new_scores = []
    for i in range(len(cards)):
        card = cards[i]
        score = scores[i]

        count = keyword_match_count(query, card)
        if count > 0:
            score = max(score + 0.15 * count, MIN_SCORE + 0.05 + 0.05 * (count - 1))

        summary = normalize_text(card.get("concern_summary", ""))
        tags = [normalize_text(t) for t in card.get("tags", [])]

        contains_price = (
            "price" in summary or
            any("price" in tag for tag in tags)
        )

        exact_add_product = (
            "add product" in summary or
            any("add product" in tag for tag in tags)
        )

        if is_add_product_query and contains_price:
            score -= 0.45

        if is_add_product_query and exact_add_product:
            score += 0.20

        new_scores.append(score)

    scores = np.array(new_scores)

    valid_indices = np.where(scores >= MIN_SCORE)[0]
    if valid_indices.size == 0:
        return []

    sorted_valid_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    top_indices = sorted_valid_indices[:TOP_K]

    return [{'card': cards[idx], 'score': float(scores[idx])} for idx in top_indices]

def has_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)

def detect_query_intent(query: str) -> str:
    q = normalize_text(query)

    if has_phrase(q, "add product") and "price" not in q:
        return "add-product"

    if "price" in q or has_phrase(q, "add price") or has_phrase(q, "product price"):
        return "pricing"

    return "generic"

def apply_intent_penalty(query: str, card: Dict[str, Any], score: float) -> float:
    intent = detect_query_intent(query)

    summary = normalize_text(card.get("concern_summary", ""))
    tags = [normalize_text(t) for t in card.get("tags", [])]

    contains_price_signal = (
        "price" in summary or
        any("price" in t for t in tags)
    )

    exact_add_product_summary = has_phrase(summary, "add product")
    exact_add_product_tag = any(has_phrase(t, "add product") for t in tags)

    if intent == "add-product":
        if contains_price_signal:
            score -= 0.45
        if exact_add_product_summary or exact_add_product_tag:
            score += 0.20

    elif intent == "pricing":
        if contains_price_signal:
            score += 0.15

    return score


# --- App Initialization ---
# NOTE: Loading the model and computing embeddings at the module level is great 
# for local development. If you deploy to production using a WSGI server like 
# Gunicorn with multiple workers, you MUST use the `--preload` flag, otherwise 
# every worker will download/load the model into RAM, causing memory crashes.

print("Loading ML model and computing embeddings... (This may take a moment)")
model = SentenceTransformer(MODEL_NAME)
cards = load_cards(JSON_PATH)
texts = [build_search_text(card) for card in cards]
card_embeddings = model.encode(texts, convert_to_numpy=True) if cards else np.array([])
print(f"Initialization complete. Loaded {len(cards)} issue cards.")


# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    results = None
    best_card = None
    best_score = None
    no_match_message = None
    exact_match = False

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            ticket_id = extract_ticket_id(query)
            matched_card = None
            if ticket_id:
                matched_card = next((c for c in cards if c.get('ticket_id', '').upper() == ticket_id), None)

            if matched_card:
                # Exact ticket ID found in the pasted text - show it directly,
                # no need to fall back to "similar" tickets for something we already have on file.
                best_card = matched_card
                best_score = 1.0
                exact_match = True
                results = [{'card': matched_card, 'score': 1.0}]
            else:
                clean_query = extract_concern(query)

                if not clean_query:
                    no_match_message = "Could not extract a concern from the input."
                else:
                    results = search_cards(clean_query, cards, card_embeddings)

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
        no_match_message=no_match_message,
        exact_match=exact_match
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)