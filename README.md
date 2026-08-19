<div align="center">

# Dashbrain

**AI-driven knowledge retrieval for support engineers.**
Paste a ticket concern, get the most similar past issue cards — no more digging through old tickets manually.

<img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/framework-Flask-black?logo=flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/inference-ONNX%20Runtime-lightgrey?logo=onnx&logoColor=white" alt="ONNX Runtime">
<img src="https://img.shields.io/badge/model-all--MiniLM--L6--v2%20(INT8)-orange" alt="Model">

</div>

<br>

Dashbrain runs on a **quantized ONNX version of `all-MiniLM-L6-v2`** for low memory usage, with sentence embeddings and cosine similarity powering the search, all served through a lightweight Flask web interface.

<br>

## 📋 How It Works

<table>
<tr>
<td width="40" align="center">1️⃣</td>
<td>Issue cards are stored in a JSON file (<code>issue_cards.json</code>), each with fields like ticket ID, module, concern summary, root cause, and recommended fix.</td>
</tr>
<tr>
<td align="center">2️⃣</td>
<td>On startup, the app embeds all issue cards into vectors and caches them (<code>card_embeddings.pkl</code>).</td>
</tr>
<tr>
<td align="center">3️⃣</td>
<td>When someone pastes a ticket/concern, the app embeds the query and finds the top matching cards via cosine similarity.</td>
</tr>
<tr>
<td align="center">4️⃣</td>
<td>Results are shown with a similarity score; exact ticket ID matches are shown directly.</td>
</tr>
</table>

<br>

## ✅ Requirements

- Python 3.12
- Windows, macOS, or Linux
- ~500MB free disk space (for dependencies + model)

<br>

## 🚀 Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/iamstevenflogio/Dashbrain.git
cd Dashbrain
```

### 2. Create a virtual environment

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
python -m venv venv
venv\Scripts\activate
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

</details>

> You should see `(venv)` appear at the start of your terminal prompt once activated.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, ONNX Runtime, Optimum, Torch (used only for building the model, not for running the app), and other required packages.

### 4. Run app.py

> Run the program. :)

```bash
python app.py
```

This downloads `sentence-transformers/all-MiniLM-L6-v2`, exports it to ONNX, and applies INT8 quantization. It creates two folders:

| Folder | Purpose |
|---|---|
| `models/dashbrain-onnx/` | The plain ONNX export |
| `models/dashbrain-int8-onnx/` | The quantized model actually used by the app |

This step needs **at least 2GB of free RAM**, since it briefly loads PyTorch and the export toolchain. It only needs to be run once, unless you change the base model or quantization settings.

> ⚠️ **Low-RAM machine?** If the script gets `Killed`, build the model on a different machine instead and copy the `models/dashbrain-int8-onnx/` folder over.

### 5. Add your issue cards

Place your `issue_cards.json` file in the project root. Each card should look like:

```json
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
```

> If this file doesn't exist yet, the app will start with zero cards until you add some.

### 6. Run the app

```bash
python app.py
```

On first run, you'll see:

```
Loading ML model...
No cache found. Embedding cards...
Initialization complete. Loaded N issue cards.
```

Subsequent runs load cached embeddings instantly instead of re-embedding everything.

### 7. Open it in your browser

```
http://127.0.0.1:5000
```

Paste a ticket concern or a full ticket ID into the search box and hit submit. 🎉

<br>

## 🌐 Letting Others On Your Network Access It

If a coworker needs to reach the app while it's running on your machine, and you're both on the same office network:

1. Make sure `app.py` runs with `host='0.0.0.0'` *(already the default in this project)*.
2. Find your machine's local IP address:
   - **Windows:** `ipconfig` → look for "IPv4 Address"
   - **macOS/Linux:** `ip addr` or `ifconfig`
3. Have them visit `http://<your-local-ip>:5000` in their browser.

<br>

## 🔄 Updating the Model or Re-embedding Cards

If you edit `issue_cards.json`, the app **automatically detects the change** and re-embeds on the next restart — no manual step needed.

If you change `quantize_model.py` (e.g., swap the base model or quantization settings), re-run it and delete the old cache so it rebuilds cleanly:

```bash
rm card_embeddings.pkl      # macOS/Linux
del card_embeddings.pkl     # Windows

python quantize_model.py
python app.py
```

<br>

## 🛠️ Troubleshooting

| Problem | Likely Cause |
|---|---|
| `NameError: name 'SentenceTransformer' is not defined` | Leftover old code in `app.py` from before the ONNX migration — check for duplicate model-loading blocks. |
| `404` on every page, including `/` | Check for a duplicate `if __name__ == '__main__':` block placed before your routes are defined — it can trigger `app.run()` too early. |
| `Killed` during `quantize_model.py` | Out of memory. Build the model on a machine with more free RAM, or add swap space, then copy the resulting `models/dashbrain-int8-onnx/` folder over. |
| `KeyError` on quantization config | Make sure you're using `AutoQuantizationConfig` from `optimum.onnxruntime.configuration`, not raw string values. |
| `ResolutionImpossible` during `pip install` | Check `requirements.txt` for conflicting `optimum` and `optimum-onnx` version pins — only one should be present. |

<br>

## 📁 Project Structure

```
Dashbrain/
├── app.py                     # Flask app + ONNX embedding wrapper + search logic
├── quantize_model.py          # One-time script to export + quantize the model
├── requirements.txt           # Python dependencies
├── issue_cards.json           # Your support ticket knowledge base (not committed)
├── card_embeddings.pkl        # Cached embeddings (auto-generated, not committed)
├── models/
│   └── dashbrain-int8-onnx/   # Quantized ONNX model (auto-generated, not committed)
└── templates/
    └── index.html             # Web UI template
```

<br>

<div align="center">

Built for support engineers who'd rather search than scroll. 🔍

</div>
