import json
import numpy as np
import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
JSON_PATH = "issue_cards.json"
TOP_K = 3


def build_search_text(card):
    tags = " ".join(card.get("tags", []))
    actions = " ".join(card.get("actions", []))
    parts = [
        card.get('concern_summary', ''),
        card.get('root_cause', ''),
        actions,
        tags
    ]
    # Filter out any empty strings and join
    return " ".join([p for p in parts if p]).strip()



def load_cards(path):
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    return [c for c in cards if c.get("ticket_id")]


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def build_embeddings(cards):
    model = load_model()
    texts = [build_search_text(card) for card in cards]
    return model.encode(texts, convert_to_numpy=True)

def extract_concern(query_text):
    # Looks for "Concern:" and grabs everything after it
    match = re.search(r'Concern:\s*(.*)', query_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return query_text.strip() # Fallback if "Concern:" isn't found

def main():
    st.title("Company Brain")

    try:
        cards = load_cards(JSON_PATH)
    except FileNotFoundError:
        st.error("issue_cards.json not found.")
        return
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return

    if not cards:
        st.warning("No valid issue cards found.")
        return

    model = load_model()
    card_embeddings = build_embeddings(cards)

    query = st.text_area("Paste ticket details")

    if st.button("Ask Company Brain"):
        # Extract just the relevant problem description
        clean_query = extract_concern(query)
    
        if not clean_query:
            st.warning("Could not extract a concern from the input.")
            return

        # Embed ONLY the clean query
        query_embedding = model.encode([clean_query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, card_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        best_idx = top_indices[0]
        best_card = cards[best_idx]
        best_score = scores[best_idx]

        st.write("Suggested Fix")

        if best_score >= 0.30:
            st.write(f"Best match: {best_card.get('ticket_id')} ({best_card.get('module')})")
            if best_card.get("root_cause"):
                st.write(f"Likely root cause: {best_card.get('root_cause')}")

            steps = best_card.get("recommended_fix") or best_card.get("actions", [])
            for step in steps:
                st.write(f"- {step}")

            st.write(f"Similarity score: {best_score:.4f}")
        else:
            st.write("No strong match found.")
            st.write(f"Best available similarity score: {best_score:.4f}")

        st.write("Top matching issue cards")

        for rank, idx in enumerate(top_indices, start=1):
            card = cards[idx]
            st.write(f"[{rank}] Ticket ID: {card.get('ticket_id')}")
            st.write(f"Lab: {card.get('lab')}")
            st.write(f"Module: {card.get('module')}")
            st.write(f"Similarity Score: {scores[idx]:.4f}")
            st.write(f"Concern: {card.get('concern_summary')}")
            st.write(f"Root Cause: {card.get('root_cause')}")
            st.write(f"Solver: {card.get('solver')}")
            st.write(f"Tags: {', '.join(card.get('tags', []))}")
            st.write("")


if __name__ == "__main__":
    main()