"""
FAISS vector store for case law similarity search.
Persists index and ID map to disk.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from config import settings

_index: faiss.IndexFlatIP | None = None
_id_map: list[int] = []          # faiss position → DB case_id
_vector_cache: dict[int, np.ndarray] = {}  # case_id → vector for rebuilding
_DIMENSION = 384                  # all-MiniLM-L6-v2 output dim


def _index_path() -> Path:
    return settings.vector_store_path() / "index.faiss"


def _map_path() -> Path:
    return settings.vector_store_path() / "id_map.json"


def _cache_path() -> Path:
    return settings.vector_store_path() / "vector_cache.npy"


def _get_index() -> faiss.IndexFlatIP:
    global _index, _id_map, _vector_cache
    if _index is not None:
        return _index

    ip = _index_path()
    mp = _map_path()
    cp = _cache_path()
    if ip.exists() and mp.exists():
        _index = faiss.read_index(str(ip))
        _id_map = json.loads(mp.read_text())
        # Rebuild vector cache from persisted file
        if cp.exists():
            _vector_cache = np.load(cp, allow_pickle=True).item()
    else:
        _index = faiss.IndexFlatIP(_DIMENSION)
        _id_map = []
    return _index


def add_vector(case_id: int, vector: np.ndarray) -> int:
    """Add a single embedding. Returns FAISS position."""
    global _vector_cache
    idx = _get_index()
    vec = vector.reshape(1, -1).astype(np.float32)
    idx.add(vec)
    pos = len(_id_map)
    _id_map.append(case_id)
    _vector_cache[case_id] = vector.copy()
    _persist()
    return pos


def add_vectors_batch(case_ids: list[int], vectors: np.ndarray) -> list[int]:
    global _vector_cache
    idx = _get_index()
    vecs = vectors.astype(np.float32)
    start = len(_id_map)
    idx.add(vecs)
    positions = list(range(start, start + len(case_ids)))
    _id_map.extend(case_ids)
    for case_id, vec in zip(case_ids, vectors):
        _vector_cache[case_id] = vec.copy()
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
    cp = _cache_path()
    faiss.write_index(_index, str(ip))
    mp.write_text(json.dumps(_id_map))
    np.save(cp, _vector_cache)


def remove_vector(case_id: int) -> bool:
    """Remove a vector by case_id. Rebuilds index without that entry."""
    global _index, _id_map, _vector_cache
    try:
        idx = _get_index()
        if case_id not in _id_map:
            return False

        # Find position of case_id
        pos = _id_map.index(case_id)

        # Rebuild index without that entry
        if idx.ntotal > 1:
            # Get all IDs except the one to remove
            all_ids = _id_map.copy()
            all_ids.pop(pos)

            # Rebuild FAISS index with retained vectors
            _index = faiss.IndexFlatIP(_DIMENSION)
            _id_map = []

            # Re-add all retained vectors from cache
            for retained_id in all_ids:
                if retained_id in _vector_cache:
                    vec = _vector_cache[retained_id].reshape(1, -1).astype(np.float32)
                    _index.add(vec)
                    _id_map.append(retained_id)

            # Remove from cache
            _vector_cache.pop(case_id, None)

            _persist()
            return True
        else:
            # If only one vector, clear the index
            _index = faiss.IndexFlatIP(_DIMENSION)
            _id_map = []
            _vector_cache.pop(case_id, None)
            _persist()
            return True
    except Exception:
        return False


def store_size() -> int:
    return _get_index().ntotal
