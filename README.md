
# Welcome to Dashbrain!

Dashbrain is a Python tool that helps support interns quickly find similar past issues and recommended fixes using a JSON knowledge base (`issue_cards.json`) a backend retrieval script (`app.py`). 

Note: (`retrieval.py`) is just a prototype without the web interface, feel free to ignore it.

## Project overview

Dashbrain acts as a lightweight "company brain" for Dashlabs.ai support work. It requires past issues to be stored as structured JSON cards and lets you retrieve similar cases to speed up investigation and responses. 

## Repository structure

```text
Dashbrain/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── .vscode/
├── issue_cards.json   # Knowledge base: list of past issues + metadata
├── app.py             # Retrieval & recommendation logic
├── retrieval.py       # Prototype retrieval script, you may ignore this
└── README.md          # Project guide and notes
```

- `issue_cards.json`  
  - Stores issue cards as JSON objects (title, description, tags, fix, notes).  
  - Extend this file whenever you resolve a new support issue.

- `retrieval.py`  
  - Loads `issue_cards.json` and provides functions or CLI commands to search for similar issues.  
  - Returns recommended cards and fix suggestions that you can adapt for tickets.

- `app.py`
  - Main application file for Dashbrain with a simple user interface.
  - An upgraded version of `retrieval.py` , using the same retrieval idea in a more usable app format.
  - Accepts issue details as input, searches the knowledge base for similar cases
  - Run this file instead of `retrieval.py`  when using Dashbrain.


