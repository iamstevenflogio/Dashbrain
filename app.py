import json
import numpy as np
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
JSON_PATH = "issue_cards.json"
TOP_K = 3
MIN_SCORE = 0.4

app = Flask(__name__)
model = SentenceTransformer(MODEL_NAME)


def build_search_text(card):
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


def load_cards(path):
    with open(path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    return [c for c in cards if c.get('ticket_id')]


def search_cards(query, cards, embeddings):
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= MIN_SCORE:
            results.append({
                'card': cards[idx],
                'score': score
            })
        if len(results) == TOP_K:
            break

    return results


cards = load_cards(JSON_PATH)
texts = [build_search_text(card) for card in cards]
card_embeddings = model.encode(texts, convert_to_numpy=True)


@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    results = None
    best_card = None
    best_score = None
    solution_note = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            results = search_cards(query, cards, card_embeddings)

            if results:
                best_card = results[0]['card']
                best_score = results[0]['score']
                solution_note = (
                    "Potential matches were found. Please review the concern summary, "
                    "root cause, recommended fix, and previous actions before applying the resolution."
                )
            else:
                solution_note = (
                    "No sufficient match was found. Please refine your query by adding the exact "
                    "issue symptoms, affected module, lab name, visible error message, status, "
                    "or troubleshooting steps already attempted."
                )

    return render_template(
        'index.html',
        query=query,
        results=results,
        best_card=best_card,
        best_score=best_score,
        solution_note=solution_note
    )

if __name__ == '__main__':
    app.run(debug=True)

