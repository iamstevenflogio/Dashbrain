Dashbrain
Dashbrain is an AI-driven knowledge retrieval tool for support engineers. Paste a ticket/concern into the app, and it returns the most similar past issue cards (with resolutions) using sentence embeddings and cosine similarity — no more digging through old tickets manually.

It runs on a quantized ONNX version of all-MiniLM-L6-v2 for low memory usage, with a Flask web interface.

How it works
Issue cards are stored in a JSON file (issue_cards.json), each with fields like ticket ID, module, concern summary, root cause, and recommended fix.

On startup, the app embeds all issue cards into vectors and caches them (card_embeddings.pkl).

When someone pastes a ticket/concern, the app embeds the query and finds the top matching cards via cosine similarity.

Results are shown with a similarity score; exact ticket ID matches are shown directly.

Requirements
Python 3.12

Windows, macOS, or Linux

~500MB free disk space (for dependencies + model)

1. Clone the repository
bash
git clone https://github.com/iamstevenflogio/Dashbrain.git
cd Dashbrain
2. Create a virtual environment
Windows (PowerShell):

powershell
python -m venv venv
venv\Scripts\activate
macOS/Linux:

bash
python3 -m venv venv
source venv/bin/activate
You should see (venv) appear at the start of your terminal prompt once activated.

3. Install dependencies
bash
pip install -r requirements.txt
This installs Flask, ONNX Runtime, Optimum, Torch (used only for building the model, not for running the app), and other required packages.

4. Build the quantized model
The ONNX model files aren't stored in this repo (they're build artifacts, not source code). Generate them locally:

bash
python quantize_model.py
This downloads sentence-transformers/all-MiniLM-L6-v2, exports it to ONNX, and applies INT8 quantization. It creates two folders:

models/dashbrain-onnx/ — the plain ONNX export

models/dashbrain-int8-onnx/ — the quantized model actually used by the app

This step needs a reasonable amount of RAM (at least 2GB free) since it briefly loads PyTorch and the export toolchain. It only needs to be run once, unless you change the base model or quantization settings.

Note: If you're setting this up on a low-RAM machine (under 2GB free) and the script gets killed, build the model on a different machine instead and copy the models/dashbrain-int8-onnx/ folder over.

5. Add your issue cards
Place your issue_cards.json file in the project root. Each card should look something like:

json
{
  "ticket_id": "SUP-1234",
  "lab": "Example Lab",
  "module": "Document Upload",
  "status": "Resolved",
  "concern_summary": "Client cannot upload documents",
  "root_cause": "File size exceeded limit",
  "recommended_fix": ["Increase upload limit", "Notify client of size cap"],
  "actions": ["Adjusted server config", "Confirmed fix with client"],
  "solver": "Jane Doe",
  "tags": ["upload", "document", "config"]
}
If this file doesn't exist yet, the app will start with zero cards until you add some.

6. Run the app
bash
python app.py
On first run, you'll see:

text
Loading ML model...
No cache found. Embedding cards...
Initialization complete. Loaded N issue cards.
Subsequent runs will load cached embeddings instantly instead of re-embedding everything.

7. Open it in your browser
Visit:

text
http://127.0.0.1:5000
Paste a ticket concern or a full ticket ID into the search box and hit submit.

Letting others on your network access it
If a coworker needs to reach the app while it's running on your machine, and you're both on the same office network:

Make sure app.py runs with host='0.0.0.0' (already the default in this project).

Find your machine's local IP address:

Windows: ipconfig (look for "IPv4 Address")

macOS/Linux: ip addr or ifconfig

Have them visit http://<your-local-ip>:5000 in their browser.

Updating the model or re-embedding cards
If you edit issue_cards.json, the app automatically detects the change and re-embeds on the next restart — no manual step needed.

If you change quantize_model.py (e.g., swap the base model or quantization settings), re-run it and delete the old cache so it rebuilds cleanly:

bash
rm card_embeddings.pkl      # macOS/Linux
del card_embeddings.pkl     # Windows
python quantize_model.py
python app.py
Troubleshooting
Problem	Likely cause
Problem	Likely cause
NameError: name 'SentenceTransformer' is not defined	Leftover old code in app.py from before the ONNX migration — check for duplicate model-loading blocks.
404 on every page, including /	Check for a duplicate if __name__ == '__main__': block placed before your routes are defined — it can trigger app.run() too early.
Killed during quantize_model.py	Out of memory. Build the model on a machine with more free RAM, or add swap space, then copy the resulting models/dashbrain-int8-onnx/ folder over.
KeyError on quantization config	Make sure you're using AutoQuantizationConfig from optimum.onnxruntime.configuration, not raw string values.
ResolutionImpossible during pip install	Check requirements.txt for conflicting optimum and optimum-onnx version pins — only one should be present.
Project structure
text
Dashbrain/
├── app.py                  # Flask app + ONNX embedding wrapper + search logic
├── quantize_model.py        # One-time script to export + quantize the model
├── requirements.txt          # Python dependencies
├── issue_cards.json          # Your support ticket knowledge base (not committed)
├── card_embeddings.pkl        # Cached embeddings (auto-generated, not committed)
├── models/
│   └── dashbrain-int8-onnx/  # Quantized ONNX model (auto-generated, not committed)
└── templates/
    └── index.html            # Web UI template
