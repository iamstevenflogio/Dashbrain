
# Welcome to Dashbrain!

Dashbrain is a Python tool that helps support interns quickly find similar past issues and recommended fixes using a JSON knowledge base (`issue_cards.json`) a backend retrieval script (`app.py`). 

## Note: (`retrieval.py`) is just a prototype without the interface, so its not connected to the main system.

## Project overview

Dashbrain acts as a lightweight "company brain" for Dashlabs.ai support work. It stores past issues as structured JSON cards and lets you retrieve similar cases to speed up investigation and responses. 

## Repository structure

```text
Dashbrain/
├─ issue_cards.json   # Knowledge base: list of past issues + metadata
├─ retrieval.py       # Retrieval & recommendation logic
└─ README.md          # Project guide and notes
```

- `issue_cards.json`  
  - Stores issue cards as JSON objects (title, description, tags, fix, notes).  
  - Extend this file whenever you resolve a new support issue.

- `retrieval.py`  
  - Loads `issue_cards.json` and provides functions or CLI commands to search for similar issues.  
  - Returns recommended cards and fix suggestions that you can adapt for tickets.

