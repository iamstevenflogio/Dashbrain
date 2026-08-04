import json
import os
import pickle
import hashlib
import re
import numpy as np
from flask import Flask, request, render_template
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
MODEL_PATH = "models/dashbrain-int8-onnx"
JSON_PATH = "issue_cards.json"
EMB_CACHE = "card_embeddings.pkl"
TOP_K = 5
MIN_SCORE = 0.50

# --- ONNX Embedding Wrapper ---
class OnnxEmbedder:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            os.path.join(model_path, "model_quantized.onnx"),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
    def encode(self, texts, normalize_embeddings=True, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="np"
            )
            onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

            outputs = self.session.run(None, onnx_inputs)
            token_embeddings = outputs[0]

            input_mask_expanded = np.expand_dims(inputs["attention_mask"], -1)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = np.clip(input_mask_expanded.sum(1), 1e-9, None)

            batch_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)

        return embeddings.astype(np.float32)

# --- Data Processing Functions ---

def build_search_text(card: Dict[str, Any]) -> str:
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
    match = re.search(r'Concern:\s*(.*)', query_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else query_text.strip()

def extract_ticket_id(query_text: str) -> Optional[str]:
    match = re.search(r'\b([A-Z]{2,6}-\d{2,6})\b', query_text.upper())
    return match.group(1) if match else None

def singularize(word: str) -> str:
    if word.endswith("ies") and len(word) > 4: return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")): return stem
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3: return word[:-1]
    return word

def normalize_text(text: str) -> str:
    words = re.findall(r'[a-z0-9\-]+', text.lower())
    return " ".join(singularize(w) for w in words)

def keyword_match_count(query: str, card: Dict[str, Any]) -> int:
    q_norm = normalize_text(query)
    count = 0
    module = card.get("module", "")
    if isinstance(module, str) and normalize_text(module) in q_norm: count += 1
    for tag in card.get("tags", []):
        if isinstance(tag, str) and normalize_text(tag) in q_norm: count += 1
    return count

def load_cards(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [c for c in data if c.get('ticket_id')]
    except json.JSONDecodeError:
        return []

def cosine_similarity(vec1, vec2):
    """Embeddings are already L2-normalized, so dot product is cosine similarity."""
    return np.dot(vec1, vec2.T)

def search_cards(query: str, cards: List[Dict], embeddings: np.ndarray) -> List[Dict[str, Any]]:
    if embeddings.size == 0: return []

    # Removed convert_to_numpy, our ONNX wrapper always returns numpy
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    new_scores = []
    for i in range(len(cards)):
        count = keyword_match_count(query, cards[i])
        if count > 0:
            boosted = max(scores[i] + 0.15 * count, MIN_SCORE + 0.05 + 0.05 * (count - 1))
            new_scores.append(boosted)
        else:
            new_scores.append(scores[i])
    scores = np.array(new_scores)

    valid_indices = np.where(scores >= MIN_SCORE)[0]
    if valid_indices.size == 0: return []

    sorted_valid_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    top_indices = sorted_valid_indices[:TOP_K]

    return [{'card': cards[idx], 'score': float(scores[idx])} for idx in top_indices]

# --- App Initialization (with Caching) ---
def model_fingerprint(model_path: str) -> str:
    h = hashlib.sha256()
    for fname in ["model_quantized.onnx", "config.json", "tokenizer_config.json"]:
        fp = os.path.join(model_path, fname)
        if os.path.exists(fp):
            h.update(fname.encode()); h.update(open(fp, 'rb').read())
    return h.hexdigest()[:16]

print("Loading ML model...")
model = OnnxEmbedder(MODEL_PATH)
cards = load_cards(JSON_PATH)

fp = model_fingerprint(MODEL_PATH)
if os.path.exists(EMB_CACHE):
    with open(EMB_CACHE, 'rb') as f:
        cache = pickle.load(f)
    if cache.get("model_fingerprint") == fp:
        card_embeddings = cache["embeddings"]
        print("Loaded cached embeddings.")
    else:
        print("Model changed. Re-embedding cards...")
        texts = [build_search_text(c) for c in cards]
        card_embeddings = model.encode(texts, normalize_embeddings=True)
        with open(EMB_CACHE, 'wb') as f:
            pickle.dump({"model_fingerprint": fp, "embeddings": card_embeddings}, f)
else:
    print("No cache found. Embedding cards...")
    texts = [build_search_text(c) for c in cards]
    card_embeddings = model.encode(texts, normalize_embeddings=True)
    with open(EMB_CACHE, 'wb') as f:
        pickle.dump({"model_fingerprint": fp, "embeddings": card_embeddings}, f)

print(f"Initialization complete. Loaded {len(cards)} issue cards.")

app = Flask(__name__)

# ... [KEEP YOUR EXISTING @app.route('/') BLOCK EXACTLY AS IT IS] ...



# --- App Initialization ---
# NOTE: Loading the model and computing embeddings at the module level is great 
# for local development. If you deploy to production using a WSGI server like 
# Gunicorn with multiple workers, you MUST use the `--preload` flag, otherwise 
# every worker will download/load the model into RAM, causing memory crashes.




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