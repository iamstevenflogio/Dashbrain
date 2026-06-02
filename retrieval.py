import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2" # BERT type model for sentence embeddings
JSON_PATH = "issue_cards.json"
TOP_K = 3   # number of similar issue cards to retrieve

def build_search_text(card):
    actions = " ".join(card.get("actions", []))
    tags = ", ".join(card.get("tags", []))
    parts = [
        f"Ticket: {card.get('ticket_id', '')}",
        f"Lab: {card.get('lab', '')}",
        f"Module: {card.get('module', '')}",
        f"Status: {card.get('status', '')}",
        f"Concern: {card.get('concern_summary', '')}",
        f"Root cause: {card.get('root_cause', '')}",
        f"Actions: {actions}",
        f"Solver: {card.get('solver', '')}",
        f"Tags: {tags}",
    ]
    return "\n".join(parts)

def load_cards(path):
    with open(path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    cards = [c for c in cards if c.get('ticket_id')]
    return cards

def main():
    cards = load_cards(JSON_PATH)
    if not cards:
        print("No valid issue cards found in issue_cards.json")
        return
    
    texts = [build_search_text(card) for card in cards]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding issue cards...")
    card_embeddings = model.encode(texts, convert_to_numpy=True)

    print(f"Loaded {len(cards)} issue cards.")
    print("Type your pasted ticket/concern below.")
    query = input("\nQuery: ").strip()

    if not query:
        print("Empty query. Exiting.")
        return
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, card_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    best_idx = top_indices[0]
    best_card = cards[best_idx]
    best_score = scores[best_idx]

    print("\n=== Suggested Fix ===\n")
    if best_score >= 0.30:
        print(f"This issue appears most similar to {best_card.get('ticket_id')} ({best_card.get('module')}).")
        print("Recommended action:")
        for step in best_card.get("actions", []):
            print(f"- {step}")

        root_cause = best_card.get("root_cause")
        if root_cause:
            print(f"\nLikely root cause: {root_cause}")
        
        print(f"\nConfidence basis: similarity score = {best_score:.4f}")
    else:
        print("No strong match found yet.")
        print("Please review the closest issue cards below before applying a fix.")
        print(f"Best available similarity score = {best_score:.4f}")
    

    print("\nTop matching issue cards:\n")
    for rank, idx in enumerate(top_indices, start=1):
        card = cards[idx]
        print(f"[{rank}] Ticket ID: {card.get('ticket_id')}")
        print(f"    Lab: {card.get('lab')}")
        print(f"    Module: {card.get('module')}")
        print(f"    Similarity Score: {scores[idx]:.4f}")
        print(f"    Concern: {card.get('concern_summary')}")
        print(f"    Root Cause: {card.get('root_cause')}")
        print(f"    Actions: {' | '.join(card.get('actions', []))}")
        print(f"    Solver: {card.get('solver')}")
        print(f"    Tags: {', '.join(card.get('tags', []))}")
        print()

if __name__ == "__main__":
    main()

