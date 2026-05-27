"""
VectorDB Engine — Python port of main.cpp

Faithful 1:1 rewrite. All HTTP endpoints, JSON shapes, demo vectors, and
algorithm behavior (HNSW / KD-Tree / BruteForce) match the C++ original so
the existing index.html frontend works unchanged.

Run:
    pip install -r requirements.txt
    python main.py
"""

import os
import math
import json
import time
import random
import threading
import heapq
from typing import Callable, List, Dict, Tuple, Optional

import requests
from flask import Flask, request, jsonify, Response, send_from_directory

# =====================================================================
#  CONFIG
# =====================================================================

DIMS = 16   # demo vector dimension
# Doc embedding dimension is determined at runtime from Ollama output


# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    __slots__ = ("id", "metadata", "category", "emb")

    def __init__(self, vid: int, metadata: str, category: str, emb: List[float]):
        self.id = vid
        self.metadata = metadata
        self.category = category
        self.emb = emb


DistFn = Callable[[List[float], List[float]], float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)


def cosine(a: List[float], b: List[float]) -> float:
    dot = na = nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += abs(a[i] - b[i])
    return s


def get_dist_fn(m: str) -> DistFn:
    if m == "cosine":
        return cosine
    if m == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort()
        return r[:k]

    def remove(self, vid: int):
        self.items = [v for v in self.items if v.id != vid]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _ins(self, n: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        if n is None:
            return KDNode(v)
        ax = d % self.dims
        if v.emb[ax] < n.item.emb[ax]:
            n.left = self._ins(n.left, v, d + 1)
        else:
            n.right = self._ins(n.right, v, d + 1)
        return n

    def insert(self, v: VectorItem):
        self.root = self._ins(self.root, v, 0)

    def _knn(self, n: Optional[KDNode], q: List[float], k: int, d: int,
             dist: DistFn, heap: list):
        # heap is a max-heap of (-distance, id) so the largest distance is at top
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, n.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, n.item.id))

        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer = n.left if diff < 0 else n.right
        farther = n.right if diff < 0 else n.left
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        r = [(-d, i) for (d, i) in heap]
        r.sort()
        return r

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class _HNSWNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.G: Dict[int, _HNSWNode] = {}
        self.rng = random.Random(seed)

    def _rand_level(self) -> int:
        u = self.rng.random()
        # guard against log(0)
        if u <= 0.0:
            u = 1e-12
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int,
                      dist: DistFn) -> List[Tuple[float, int]]:
        vis = {ep: True}
        # min-heap of (distance, id) for candidates to explore
        cands: List[Tuple[float, int]] = []
        # max-heap (stored as negated distance) for found set
        found_max: List[Tuple[float, int]] = []

        d0 = dist(q, self.G[ep].item.emb)
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found_max, (-d0, ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            # if our closest candidate is worse than the worst in found, stop
            if len(found_max) >= ef and cd > -found_max[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if vis.get(nid) or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist(q, self.G[nid].item.emb)
                if len(found_max) < ef or nd < -found_max[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found_max, (-nd, nid))
                    if len(found_max) > ef:
                        heapq.heappop(found_max)

        res = [(-d, i) for (d, i) in found_max]
        res.sort()
        return res

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        vid = item.id
        lvl = self._rand_level()
        self.G[vid] = _HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = vid
            self.top_layer = lvl
            return

        ep = self.entry_pt
        # zoom from top layer down to lvl+1, greedily following nearest
        lc = self.top_layer
        while lc > lvl:
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
            lc -= 1

        # insert from min(top_layer, lvl) down to 0
        lc = min(self.top_layer, lvl)
        while lc >= 0:
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            self.G[vid].nbrs[lc] = list(sel)

            for nid in sel:
                neigh = self.G.get(nid)
                if neigh is None:
                    continue
                if len(neigh.nbrs) <= lc:
                    # extend with empty layers
                    neigh.nbrs.extend([[] for _ in range(lc + 1 - len(neigh.nbrs))])
                conn = neigh.nbrs[lc]
                conn.append(vid)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist(neigh.item.emb, self.G[c].item.emb), c))
                    ds.sort()
                    neigh.nbrs[lc] = [d[1] for d in ds[:max_m]]
            if W:
                ep = W[0][1]
            lc -= 1

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = vid

    def knn(self, q: List[float], k: int, ef: int,
            dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        lc = self.top_layer
        while lc > 0:
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
            lc -= 1
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, vid: int):
        if vid not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                # remove all occurrences
                while vid in layer:
                    layer.remove(vid)
        if self.entry_pt == vid:
            self.entry_pt = -1
            for nid in self.G:
                if nid != vid:
                    self.entry_pt = nid
                    break
        del self.G[vid]

    def get_info(self) -> dict:
        node_count = len(self.G)
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_out = []
        edges_out = []
        for vid, nd in self.G.items():
            nodes_out.append({
                "id": vid,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if vid < nid:
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": vid, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": node_count,
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes_out,
            "edges": edges_out,
        }

    def size(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(m=16, ef_build=200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, list(emb))
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, vid: int) -> bool:
        with self.mu:
            if vid not in self.store:
                return False
            del self.store[vid]
            self.bf.remove(vid)
            self.hnsw.remove(vid)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, vid in raw:
                if vid in self.store:
                    it = self.store[vid]
                    hits.append({
                        "id": it.id,
                        "metadata": it.metadata,
                        "category": it.category,
                        "distance": round(float(d), 6),
                        "embedding": [round(float(x), 4) for x in it.emb],
                    })
            return {
                "results": hits,
                "latencyUs": us,
                "algo": algo,
                "metric": metric,
            }

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)

            def measure(fn) -> int:
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": measure(lambda: self.bf.knn(q, k, dfn)),
                "kdtreeUs": measure(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": measure(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "itemCount": len(self.store),
            }

    def all(self) -> List[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: List[str] = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            r = requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=(3, 30),
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=(3, 180),
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

class DocItem:
    __slots__ = ("id", "title", "text", "emb")

    def __init__(self, did: int, title: str, text: str, emb: List[float]):
        self.id = did
        self.title = title
        self.text = text
        self.emb = emb


class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(m=16, ef_build=200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, list(emb))
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", list(emb))
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int,
               max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d, did in raw:
                if did in self.store and d <= max_dist:
                    out.append((d, self.store[did]))
            return out

    def remove(self, did: int) -> bool:
        with self.mu:
            if did not in self.store:
                return False
            del self.store[did]
            self.hnsw.remove(did)
            self.bf.remove(did)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# =====================================================================
#  DEMO DATA  (same 20 hand-crafted 16D vectors as main.cpp)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    demo = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demo:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  HTTP SERVER  (Flask — mirrors the cpp-httplib routes in main.cpp)
# =====================================================================

app = Flask(__name__)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


def _cors(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.after_request
def _after(resp):
    return _cors(resp)


@app.route("/<path:_p>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def _options(_p=None):
    return ("", 204)


def _parse_vec(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


# ── DEMO VECTOR ENDPOINTS ───────────────────────────────────────────

@app.get("/search")
def http_search():
    q = _parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    try:
        k = int(request.args.get("k", "5"))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    return jsonify(db.search(q, k, metric, algo))


@app.post("/insert")
def http_insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or not isinstance(emb, list) or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})
    try:
        emb = [float(x) for x in emb]
    except (TypeError, ValueError):
        return jsonify({"error": "invalid body"})
    vid = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": vid})


@app.delete("/delete/<int:vid>")
def http_delete(vid: int):
    ok = db.remove(vid)
    return jsonify({"ok": ok})


@app.get("/items")
def http_items():
    items = db.all()
    return jsonify([
        {
            "id": v.id,
            "metadata": v.metadata,
            "category": v.category,
            "embedding": [round(float(x), 4) for x in v.emb],
        }
        for v in items
    ])


@app.get("/benchmark")
def http_benchmark():
    q = _parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    try:
        k = int(request.args.get("k", "5"))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    return jsonify(db.benchmark(q, k, metric))


@app.get("/hnsw-info")
def http_hnsw_info():
    return jsonify(db.hnsw_info())


# ── DOCUMENT + RAG ENDPOINTS ────────────────────────────────────────

@app.post("/doc/insert")
def http_doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids: List[int] = []
    total = len(chunks)
    for i, ch in enumerate(chunks):
        emb = ollama.embed(ch)
        if not emb:
            return jsonify({"error":
                "Ollama unavailable. Install from https://ollama.com then run: "
                "ollama pull nomic-embed-text && ollama pull llama3.2"})
        chunk_title = f"{title} [{i+1}/{total}]" if total > 1 else title
        ids.append(doc_db.insert(chunk_title, ch, emb))

    return jsonify({"ids": ids, "chunks": total, "dims": doc_db.get_dims()})


@app.delete("/doc/delete/<int:did>")
def http_doc_delete(did: int):
    ok = doc_db.remove(did)
    return jsonify({"ok": ok})


@app.get("/doc/list")
def http_doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        words = d.text.count(" ") + 1
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": words,
        })
    return jsonify(out)


@app.post("/doc/search")
def http_doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    try:
        k = int(body.get("k", 3))
    except (TypeError, ValueError):
        k = 3
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [
            {
                "id": d.id,
                "title": d.title,
                "distance": round(float(dist), 4),
            }
            for dist, d in hits
        ]
    })


@app.post("/doc/ask")
def http_doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    try:
        k = int(body.get("k", 3))
    except (TypeError, ValueError):
        k = 3
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)

    ctx_parts = []
    for i, (dist, d) in enumerate(hits):
        ctx_parts.append(f"[{i+1}] {d.title}:\n{d.text}\n\n")
    ctx = "".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    answer = ollama.generate(prompt)

    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {
                "id": d.id,
                "title": d.title,
                "text": d.text,
                "distance": round(float(dist), 4),
            }
            for dist, d in hits
        ],
        "docCount": doc_db.size(),
    })


@app.get("/status")
def http_status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.get("/stats")
def http_stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


# ── Serve index.html at root ────────────────────────────────────────

@app.get("/")
def http_index():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "index.html")):
        return send_from_directory(here, "index.html")
    return ("index.html not found in script directory", 404)


# =====================================================================
#  MAIN
# =====================================================================

def _banner():
    up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama:", "ONLINE" if up else "OFFLINE (install from ollama.com)")
    if up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")


if __name__ == "__main__":
    _banner()
    # threaded=True so concurrent requests (e.g. RAG generate while UI polls) work
    app.run(host="0.0.0.0", port=8080, threaded=True, debug=False)
