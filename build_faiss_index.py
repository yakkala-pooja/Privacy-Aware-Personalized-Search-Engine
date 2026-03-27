import numpy as np
import faiss
import json
import os
import time
from tqdm import tqdm

class FAISSIndexBuilder:
    """
    Build and manage FAISS index for efficient similarity search.
    Uses IndexIVFPQ with IVF clusters and Product Quantization.
    """
    
    def __init__(self, dimension=768, n_clusters=100, n_subvectors=8, n_bits=8):
        """
        Initialize FAISS index builder.
        
        Args:
            dimension (int): Dimension of the vectors
            n_clusters (int): Number of IVF clusters
            n_subvectors (int): Number of PQ sub-vectors
            n_bits (int): Number of bits per PQ sub-vector
        """
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.n_subvectors = n_subvectors
        self.n_bits = n_bits
        
        # Check for GPU availability
        self.use_gpu = self._check_gpu_availability()
        print(f"GPU acceleration: {'Available' if self.use_gpu else 'Not available'}")
        
        # Initialize index
        self.index = None
        self.is_trained = False
    
    def _check_gpu_availability(self):
        """Check if GPU resources are available for FAISS."""
        try:
            # Try to get number of GPUs
            ngpus = faiss.get_num_gpus()
            print(f"Found {ngpus} GPU(s)")
            return ngpus > 0
        except:
            print("No GPU support available, using CPU")
            return False
    
    def build_index(self, vectors, index_dir="faiss_index"):
        """
        Build and train the FAISS index.
        
        Args:
            vectors (np.ndarray): Input vectors to index
            index_dir (str): Directory to save the index
        
        Returns:
            faiss.IndexIVFPQ: Trained FAISS index
        """
        print(f"Building FAISS index for {len(vectors)} vectors...")
        print(f"Vector dimension: {vectors.shape[1]}")
        print(f"IVF clusters: {self.n_clusters}")
        print(f"PQ sub-vectors: {self.n_subvectors}")
        print(f"PQ bits per sub-vector: {self.n_bits}")
        
        # Ensure vectors are float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Create the coarse quantizer (flat index for IVF)
        coarse_quantizer = faiss.IndexFlatIP(self.dimension)
        
        # Create the IVF index with PQ compression
        self.index = faiss.IndexIVFPQ(
            coarse_quantizer, 
            self.dimension, 
            self.n_clusters, 
            self.n_subvectors, 
            self.n_bits
        )
        
        # Train the index
        print("Training the index...")
        start_time = time.time()
        
        # Use a subset of vectors for training
        n_train = min(100000, len(vectors))
        train_vectors = vectors[:n_train]
        
        self.index.train(train_vectors)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Add all vectors to the index
        print("Adding vectors to the index...")
        start_time = time.time()
        
        # Add vectors in batches to manage memory
        batch_size = 10000
        for i in tqdm(range(0, len(vectors), batch_size), desc="Adding vectors"):
            batch_vectors = vectors[i:i + batch_size]
            self.index.add(batch_vectors)
        
        adding_time = time.time() - start_time
        print(f"Adding vectors completed in {adding_time:.2f} seconds")
        
        self.is_trained = True
        
        # Save the index
        self.save_index(index_dir)
        
        return self.index
    
    def save_index(self, index_dir):
        """Save the trained index to disk."""
        os.makedirs(index_dir, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(index_dir, "passage_index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "n_clusters": self.n_clusters,
            "n_subvectors": self.n_subvectors,
            "n_bits": self.n_bits,
            "total_vectors": self.index.ntotal,
            "is_trained": self.is_trained,
            "use_gpu": self.use_gpu
        }
        
        metadata_path = os.path.join(index_dir, "index_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Get index size
        index_size = os.path.getsize(index_path) / (1024 * 1024)  # MB
        print(f"Index saved to {index_path}")
        print(f"Index size: {index_size:.2f} MB")
        
        return index_path
    
    def load_index(self, index_dir="faiss_index"):
        """Load a saved index from disk."""
        index_path = os.path.join(index_dir, "passage_index.faiss")
        metadata_path = os.path.join(index_dir, "index_metadata.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load the index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded index with {metadata['total_vectors']} vectors")
        
        self.is_trained = True
        return self.index
    
    def search(self, query_vectors, k=10, nprobe=10):
        """
        Search the index for similar vectors.
        
        Args:
            query_vectors (np.ndarray): Query vectors
            k (int): Number of nearest neighbors to retrieve
            nprobe (int): Number of clusters to probe during search
        
        Returns:
            tuple: (distances, indices)
        """
        if not self.is_trained:
            raise ValueError("Index must be trained before searching")
        
        # Set nprobe for search
        self.index.nprobe = nprobe
        
        # Ensure query vectors are float32
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        
        # Perform search
        distances, indices = self.index.search(query_vectors, k)
        
        return distances, indices
    
    def get_index_stats(self):
        """Get statistics about the index."""
        if not self.is_trained:
            return None
        
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "n_clusters": self.index.nlist,
            "n_subvectors": self.index.pq.M,
            "n_bits": self.index.pq.nbits,
            "is_trained": self.is_trained,
            "use_gpu": self.use_gpu
        }
        
        return stats

def build_passage_index(embeddings_dir="embeddings", index_dir="faiss_index"):
    """
    Build FAISS index for passage embeddings.
    
    Args:
        embeddings_dir (str): Directory containing embeddings
        index_dir (str): Directory to save the index
    """
    # Load passage embeddings
    print(f"Loading passage embeddings from {embeddings_dir}...")
    passage_embeddings = np.load(os.path.join(embeddings_dir, "passage_embeddings.npy"))
    
    print(f"Passage embeddings shape: {passage_embeddings.shape}")
    
    # Initialize index builder
    index_builder = FAISSIndexBuilder(
        dimension=768,
        n_clusters=100,
        n_subvectors=8,
        n_bits=8
    )
    
    # Build the index
    index = index_builder.build_index(passage_embeddings, index_dir)
    
    # Get index statistics
    stats = index_builder.get_index_stats()
    print(f"\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return index_builder

def test_index_search(index_builder, embeddings_dir="embeddings", n_queries=10):
    """
    Test the index with sample queries.
    
    Args:
        index_builder (FAISSIndexBuilder): Trained index builder
        embeddings_dir (str): Directory containing embeddings
        n_queries (int): Number of test queries
    """
    # Load query embeddings for testing
    query_embeddings = np.load(os.path.join(embeddings_dir, "query_embeddings.npy"))
    
    # Use first n_queries as test queries
    test_queries = query_embeddings[:n_queries]
    
    print(f"\nTesting index search with {n_queries} queries...")
    start_time = time.time()
    
    # Search for similar passages
    distances, indices = index_builder.search(test_queries, k=5, nprobe=10)
    
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.4f} seconds")
    print(f"Average search time per query: {search_time/n_queries:.4f} seconds")
    
    # Print sample results
    print(f"\nSample search results:")
    for i in range(min(3, n_queries)):
        print(f"Query {i+1}:")
        for j in range(5):
            print(f"  Result {j+1}: Index {indices[i][j]}, Distance {distances[i][j]:.4f}")
        print()

if __name__ == "__main__":
    # Build the FAISS index
    print("Building FAISS index for passage embeddings...")
    index_builder = build_passage_index()
    
    # Test the index
    test_index_search(index_builder)
    
    print("FAISS index construction completed successfully!") 