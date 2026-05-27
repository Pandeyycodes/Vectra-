# Setup Guide — Vectra

Get the project running on your machine in 5 steps.

---

## What You Need

| Tool | Version | Download |
|---|---|---|
| Python | 3.9 or newer | https://www.python.org/downloads/ |
| Git | Any | https://git-scm.com/download/win |
| Ollama | Latest | https://ollama.com |

> **Minimum specs:** 8GB RAM, ~3GB free disk space for the AI models.

---

## Step 1 — Install Python

1. Go to https://www.python.org/downloads/ and download Python 3.9+
2. Run the installer
3. **Critical:** on the first screen, tick **"Add python.exe to PATH"** before clicking Install
4. Open a new PowerShell and verify:

```powershell
python --version
pip --version
```

Both commands should return a version number. If they don't, reinstall and make sure you ticked "Add to PATH".

---

## Step 2 — Install Git

1. Go to https://git-scm.com/download/win
2. Download and run the installer with default settings
3. Verify:

```powershell
git --version
```

---

## Step 3 — Install Ollama and Pull Models

Ollama runs the AI models locally on your machine — no API keys, no internet required after setup.

1. Go to https://ollama.com and click **Download for Windows**
2. Run the installer — Ollama starts automatically in your system tray
3. Open PowerShell and pull the two required models:

```powershell
ollama pull nomic-embed-text
```
*~274 MB — converts your text into 768-dimensional vectors*

```powershell
ollama pull llama3.2
```
*~2 GB — the language model that answers your questions*

4. Verify both downloaded:

```powershell
ollama list
```

You should see both `nomic-embed-text` and `llama3.2` listed.

---

## Step 4 — Clone and Install Dependencies

```powershell
git clone https://github.com/Pandeyycodes/Vectra-.git
cd Vectra-
pip install -r requirements.txt
```

`requirements.txt` only has two packages — `flask` and `requests`. Takes under 30 seconds.

---

## Step 5 — Run

**Terminal 1** — start Ollama (skip if it's already running in your system tray):
```powershell
ollama serve
```

**Terminal 2** — start Vectra:
```powershell
python main.py
```

You should see:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open your browser and go to **http://localhost:8080**

---

## Using the App

### Tab 1 — Search
Type any concept (`binary tree`, `sushi`, `basketball`, `calculus`) and hit **Run search**. The scatter plot highlights the nearest vectors in real time. Use **Compare all algorithms** to benchmark HNSW vs KD-Tree vs Brute Force side by side.

### Tab 2 — Documents
Paste any text with a title and click **Embed & insert**. Long documents are automatically split into overlapping 250-word chunks — each chunk gets its own 768D embedding and appears as a new point on the scatter plot.

### Tab 3 — Ask AI
Ask any question about the documents you inserted. Vectra embeds your question, runs HNSW to find the most relevant chunks, and passes them to llama3.2 which generates an answer. The retrieved context chunks are shown below the answer.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in the header | Run `ollama serve` in a terminal |
| Embedding takes a long time | First run downloads the model — wait 2 minutes |
| `python: command not found` | Reinstall Python and tick "Add to PATH" |
| `ModuleNotFoundError: flask` | Run `pip install -r requirements.txt` |
| Port 8080 already in use | Run `netstat -ano \| findstr 8080`, then `taskkill /PID <pid> /F` |
| LLM answers are slow | Normal on CPU — switch to the smaller model (see below) |

### Switch to a Faster Model

If `llama3.2` is too slow on your machine, use the 1B version:

```powershell
ollama pull llama3.2:1b
```

Then open `main.py`, find the `OllamaClient` class and change:

```python
self.gen_model = "llama3.2:1b"
```

Restart the server.

---

## REST API

Everything the frontend does is also available via the API at `http://localhost:8080`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | KNN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `POST` | `/doc/insert` | Embed and store a document |
| `GET` | `/doc/list` | List all stored documents |
| `DELETE` | `/doc/delete/:id` | Delete a document chunk |
| `POST` | `/doc/ask` | RAG: retrieve context and generate answer |
| `GET` | `/status` | Ollama status and model info |

### Example — search via curl:

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example — ask a question via curl:

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```
