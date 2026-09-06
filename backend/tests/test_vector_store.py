"""
Regression tests for vector_store.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss
from rag import vector_store


def test_middle_vector_removal():
    """Test removing a middle vector maintains FAISS position alignment."""
    # Save original state
    original_index = vector_store._index
    original_id_map = vector_store._id_map.copy()
    original_vector_cache = vector_store._vector_cache.copy()
    original_persist = vector_store._persist
    
    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass
    
    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()
        
        # Mock persistence
        vector_store._persist = mock_persist
        
        # Add three vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        
        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)
        
        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]
        
        # Remove middle vector (case_id 2)
        result = vector_store.remove_vector(2)
        assert result is True
        
        # Verify store_size equals len(_id_map)
        assert vector_store.store_size() == len(vector_store._id_map), f"store_size={vector_store.store_size()}, len(_id_map)={len(vector_store._id_map)}"
        assert vector_store.store_size() == 2
        assert len(vector_store._id_map) == 2
        assert vector_store._id_map == [1, 3], f"Expected [1, 3], got {vector_store._id_map}"
        
        # Verify searches resolve to correct case IDs
        query_vec = vec1
        results = vector_store.search(query_vec, top_k=2)
        result_ids = [r["case_id"] for r in results]
        assert 1 in result_ids, "Case ID 1 should be in search results"
        assert 3 in result_ids, "Case ID 3 should be in search results"
        assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"
        
    finally:
        # Restore original state
        vector_store._persist = original_persist
        vector_store._index = original_index
        vector_store._id_map = original_id_map
        vector_store._vector_cache = original_vector_cache


def test_cache_preservation_after_rebuild():
    """Test that vector cache is used correctly during index rebuild in remove_vector."""
    # Save original state
    original_index = vector_store._index
    original_id_map = vector_store._id_map.copy()
    original_vector_cache = vector_store._vector_cache.copy()
    original_persist = vector_store._persist
    
    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass
    
    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()
        
        # Mock persistence
        vector_store._persist = mock_persist
        
        # Add multiple vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        
        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)
        
        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]
        assert len(vector_store._vector_cache) == 3
        
        # Remove middle vector - this triggers cache-based rebuild
        result = vector_store.remove_vector(2)
        assert result is True
        
        # Verify removal worked correctly using cache
        assert vector_store.store_size() == 2
        assert len(vector_store._id_map) == 2
        assert vector_store._id_map == [1, 3]
        assert 2 not in vector_store._vector_cache
        assert 1 in vector_store._vector_cache
        assert 3 in vector_store._vector_cache
        
        # Verify search results are retained for remaining cases
        query_vec = vec1
        results = vector_store.search(query_vec, top_k=2)
        result_ids = [r["case_id"] for r in results]
        assert 1 in result_ids, "Case ID 1 should be in search results"
        assert 3 in result_ids, "Case ID 3 should be in search results"
        assert 2 not in result_ids, "Case ID 2 should not be in search results after removal"
        
        # Verify store_size() is consistent
        assert vector_store.store_size() == len(vector_store._id_map)
        
    finally:
        # Restore original state
        vector_store._persist = original_persist
        vector_store._index = original_index
        vector_store._id_map = original_id_map
        vector_store._vector_cache = original_vector_cache


def test_missing_cache_file():
    """Test that vector cache is reconstructed from FAISS index when cache file is missing."""
    # Save original state
    original_index = vector_store._index
    original_id_map = vector_store._id_map.copy()
    original_vector_cache = vector_store._vector_cache.copy()
    original_persist = vector_store._persist

    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass

    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()

        # Mock persistence
        vector_store._persist = mock_persist

        # Add multiple vectors
        vec1 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec2 = np.random.rand(vector_store._DIMENSION).astype(np.float32)
        vec3 = np.random.rand(vector_store._DIMENSION).astype(np.float32)

        vector_store.add_vector(1, vec1)
        vector_store.add_vector(2, vec2)
        vector_store.add_vector(3, vec3)

        # Verify initial state
        assert vector_store.store_size() == 3
        assert len(vector_store._id_map) == 3
        assert vector_store._id_map == [1, 2, 3]
        assert len(vector_store._vector_cache) == 3

        # Simulate missing cache file by clearing the cache
        vector_store._vector_cache.clear()

        # Call the reconstruction function directly
        vector_store._reconstruct_cache_from_index()

        # Verify cache was reconstructed from FAISS index
        assert len(vector_store._vector_cache) == 3, "Cache should be reconstructed from FAISS index"
        assert 1 in vector_store._vector_cache
        assert 2 in vector_store._vector_cache
        assert 3 in vector_store._vector_cache

        # Verify reconstructed vectors match original vectors
        assert np.allclose(vector_store._vector_cache[1], vec1)
        assert np.allclose(vector_store._vector_cache[2], vec2)
        assert np.allclose(vector_store._vector_cache[3], vec3)

        # Verify that removal still works after cache reconstruction
        result = vector_store.remove_vector(2)
        assert result is True
        assert vector_store.store_size() == 2
        assert vector_store._id_map == [1, 3]

    finally:
        # Restore original state
        vector_store._persist = original_persist
        vector_store._index = original_index
        vector_store._id_map = original_id_map
        vector_store._vector_cache = original_vector_cache


if __name__ == "__main__":
    test_middle_vector_removal()
    print("Test passed: middle_vector_removal")
    test_cache_preservation_after_rebuild()
    print("Test passed: cache_preservation_after_rebuild")
    test_missing_cache_file()
    print("Test passed: missing_cache_file")
