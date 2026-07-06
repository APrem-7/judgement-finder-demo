"""
FAISS vector store for case law similarity search.
Persists index and ID map to disk.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from ..config import settings

_index: faiss.IndexFlatIP | None = None
_id_map: list[int] = []          # faiss position → DB case_id
_DIMENSION = 384                  # all-MiniLM-L6-v2 output dim


def _index_path() -> Path:
    return settings.vector_store_path() / "index.faiss"


def _map_path() -> Path:
    return settings.vector_store_path() / "id_map.json"


def _get_index() -> faiss.IndexFlatIP:
    global _index, _id_map
    if _index is not None:
        return _index

    ip = _index_path()
    mp = _map_path()
    if ip.exists() and mp.exists():
        _index = faiss.read_index(str(ip))
        _id_map = json.loads(mp.read_text())
    else:
        _index = faiss.IndexFlatIP(_DIMENSION)
        _id_map = []
    return _index


def add_vector(case_id: int, vector: np.ndarray) -> int:
    """Add a single embedding. Returns FAISS position."""
    idx = _get_index()
    vec = vector.reshape(1, -1).astype(np.float32)
    idx.add(vec)
    pos = len(_id_map)
    _id_map.append(case_id)
    _persist()
    return pos


def add_vectors_batch(case_ids: list[int], vectors: np.ndarray) -> list[int]:
    idx = _get_index()
    vecs = vectors.astype(np.float32)
    start = len(_id_map)
    idx.add(vecs)
    positions = list(range(start, start + len(case_ids)))
    _id_map.extend(case_ids)
    _persist()
    return positions


def search(query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
    """Returns list of {case_id, score} sorted by descending similarity."""
    idx = _get_index()
    if idx.ntotal == 0:
        return []
    vec = query_vector.reshape(1, -1).astype(np.float32)
    k = min(top_k, idx.ntotal)
    scores, positions = idx.search(vec, k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        results.append({"case_id": _id_map[pos], "score": float(round(score, 4))})
    return results


def _persist():
    ip = _index_path()
    mp = _map_path()
    faiss.write_index(_index, str(ip))
    mp.write_text(json.dumps(_id_map))


def store_size() -> int:
    return _get_index().ntotal
