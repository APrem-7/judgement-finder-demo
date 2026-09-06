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


def test_interrupted_write_consistency():
    """Test that interrupted writes don't cause mismatched case IDs in search results."""
    import shutil

    # Save original state
    original_index = vector_store._index
    original_id_map = vector_store._id_map.copy()
    original_vector_cache = vector_store._vector_cache.copy()
    original_persist = vector_store._persist
    original_generation = vector_store._generation

    # Mock _persist to prevent disk I/O during test
    def mock_persist():
        pass

    try:
        # Reset global state
        vector_store._index = faiss.IndexFlatIP(vector_store._DIMENSION)
        vector_store._id_map.clear()
        vector_store._vector_cache.clear()
        vector_store._generation = 0

        # Mock persistence during setup
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

        # Restore original persist to test actual persistence
        vector_store._persist = original_persist

        # Perform a successful persist to create generation 1
        vector_store._persist()
        assert vector_store._generation == 1

        # Simulate interrupted write by creating a partial generation 2
        gen2_dir = vector_store._generation_dir(2)
        gen2_dir.mkdir(parents=True, exist_ok=True)
        
        # Write only the index (simulating interruption before id_map write)
        gen2_index = vector_store._index_path(2)
        faiss.write_index(vector_store._index, str(gen2_index))
        
        # Write a manifest pointing to generation 2 (simulating interrupted atomic update)
        manifest_path = vector_store._manifest_path()
        temp_manifest = manifest_path.with_suffix('.tmp')
        temp_manifest.write_text('{"generation": 2}')
        temp_manifest.replace(manifest_path)

        # Now force reload by clearing in-memory state
        vector_store._index = None
        vector_store._id_map = []
        vector_store._vector_cache = {}
        vector_store._generation = 0

        # Reload should detect the incomplete generation and fall back to generation 1
        idx = vector_store._get_index()
        
        # Since generation 2 is incomplete (missing id_map.json), it should fall back
        # The system should either load gen 1 or create a new index
        # In either case, verify consistency
        
        # Perform search and verify no mismatched case IDs
        if vector_store.store_size() > 0:
            query_vec = vec1
            results = vector_store.search(query_vec, top_k=3)
            result_ids = [r["case_id"] for r in results]
            
            # All result IDs should be valid integers
            for result_id in result_ids:
                assert isinstance(result_id, int), f"Case ID should be int, got {type(result_id)}"
            
            # The id_map length should match the store size
            assert len(vector_store._id_map) == vector_store.store_size(), \
                f"ID map length {len(vector_store._id_map)} doesn't match store size {vector_store.store_size()}"

        # Clean up the interrupted generation
        if gen2_dir.exists():
            shutil.rmtree(gen2_dir)

    finally:
        # Restore original state
        vector_store._persist = original_persist
        vector_store._index = original_index
        vector_store._id_map = original_id_map
        vector_store._vector_cache = original_vector_cache
        vector_store._generation = original_generation


if __name__ == "__main__":
    test_middle_vector_removal()
    print("Test passed: middle_vector_removal")
    test_cache_preservation_after_rebuild()
    print("Test passed: cache_preservation_after_rebuild")
    test_missing_cache_file()
    print("Test passed: missing_cache_file")
    test_interrupted_write_consistency()
    print("Test passed: interrupted_write_consistency")
