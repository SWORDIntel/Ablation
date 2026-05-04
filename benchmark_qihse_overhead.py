import time
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from aegis_lab.state.qihse_wrapper import QIHSE, QihseVectorDBBackend

def benchmark_qihse():
    # We need a dummy library or a real one. 
    # Since I don't have the real .so easily accessible in the right path for Python, 
    # I might need to find where it is or mock it if I just want to test Python overhead.
    # Looking at the file system, /home/john/Ablation/QIHSE/qihse/libqihse.so exists.
    
    lib_path = "/home/john/Ablation/QIHSE/qihse/libqihse.so"
    if not os.path.exists(lib_path):
        print(f"Library not found at {lib_path}")
        return

    q = QIHSE(lib_path)
    db = q.create_vector_db(QihseVectorDBBackend.INMEMORY)
    
    num_vectors = 10000
    dims = 128
    vectors = np.random.rand(num_vectors, dims).astype(np.float32).tolist()
    metadata = [f"meta_{i}" for i in range(num_vectors)]
    
    print(f"Benchmarking add_to_vector_db with {num_vectors} vectors of {dims} dims...")
    start = time.time()
    q.add_to_vector_db(db, vectors, metadata)
    end = time.time()
    print(f"add_to_vector_db took: {end - start:.4f}s")
    
    query_vector = np.random.rand(dims).astype(np.float32).tolist()
    print(f"Benchmarking search_vector_db...")
    start = time.time()
    for _ in range(100):
        q.search_vector_db(db, query_vector, top_k=10)
    end = time.time()
    print(f"search_vector_db (100 calls) took: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_qihse()
