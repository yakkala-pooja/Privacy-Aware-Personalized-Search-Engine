import pandas as pd
import numpy as np
from generate_embeddings import DistilBERTEmbedder, load_and_embed_dataset

def test_embedding_generation():
    """Test the embedding generation with a small sample."""
    
    print("Testing DistilBERT embedding generation...")
    
    # Load a small sample of the dataset
    df = pd.read_csv("marco_preprocessed.csv")
    sample_df = df.head(100)  # Use first 100 samples for testing
    
    print(f"Testing with {len(sample_df)} samples")
    
    # Initialize embedder
    embedder = DistilBERTEmbedder()
    
    # Test with a few sample texts
    test_queries = ["what is machine learning", "how to cook pasta", "best programming languages"]
    test_passages = [
        "Machine learning is a subset of artificial intelligence.",
        "To cook pasta, bring water to boil and add pasta.",
        "Popular programming languages include Python and JavaScript."
    ]
    
    print("\nTesting query embeddings...")
    query_embeddings = embedder.generate_query_embeddings(test_queries)
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Query embedding dimension: {query_embeddings.shape[1]}")
    
    print("\nTesting passage embeddings...")
    passage_embeddings = embedder.generate_passage_embeddings(test_passages)
    print(f"Passage embeddings shape: {passage_embeddings.shape}")
    print(f"Passage embedding dimension: {passage_embeddings.shape[1]}")
    
    # Test similarity computation
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\nTesting similarity computation...")
    similarities = cosine_similarity(query_embeddings, passage_embeddings)
    print(f"Similarity matrix shape: {similarities.shape}")
    print("Similarity scores:")
    for i, query in enumerate(test_queries):
        for j, passage in enumerate(test_passages):
            print(f"  Query '{query}' vs Passage '{passage[:50]}...': {similarities[i][j]:.4f}")
    
    print("\nEmbedding generation test completed successfully!")
    return True

if __name__ == "__main__":
    test_embedding_generation() 