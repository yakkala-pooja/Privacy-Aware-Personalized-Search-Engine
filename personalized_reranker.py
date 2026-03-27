import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import json
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

class PersonalizedReranker(nn.Module):
    """
    Multi-head attention-based personalized reranker.
    Uses attention to reweight passage vectors based on user profile.
    """
    
    def __init__(self, embedding_dim=768, num_heads=4, num_layers=1, dropout=0.1):
        super(PersonalizedReranker, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention for user-passage interaction
        self.user_passage_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network for final scoring
        self.scoring_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Device
        self.device = torch.device('cpu')
        self.to(self.device)
        self.eval()
    
    def forward(self, user_profile, passage_embeddings, original_scores=None):
        """
        Forward pass for personalized reranking.
        
        Args:
            user_profile (torch.Tensor): User profile embedding (batch_size, embedding_dim)
            passage_embeddings (torch.Tensor): Passage embeddings (batch_size, num_passages, embedding_dim)
            original_scores (torch.Tensor): Original FAISS similarity scores (batch_size, num_passages)
        
        Returns:
            torch.Tensor: Personalized scores (batch_size, num_passages)
        """
        batch_size, num_passages, embedding_dim = passage_embeddings.shape
        
        # Expand user profile to match passage sequence length
        user_profile_expanded = user_profile.unsqueeze(1).expand(-1, num_passages, -1)
        
        # Create sequence: [user_profile, passage_1, passage_2, ..., passage_n]
        sequence = torch.cat([user_profile_expanded, passage_embeddings], dim=1)
        
        # Multi-head attention between user profile and passages
        attended, attention_weights = self.user_passage_attention(
            sequence, sequence, sequence
        )
        
        # Residual connection and normalization
        attended = self.norm1(sequence + self.dropout(attended))
        
        # Take the user profile position (index 0) as the reweighted user representation
        reweighted_user = attended[:, 0, :]  # Shape: (batch_size, embedding_dim)
        
        # Calculate attention-weighted similarity for each passage
        attention_scores = []
        
        for i in range(num_passages):
            # Get the attention weights for this passage (index i+1 in sequence)
            # attention_weights shape: (batch_size, seq_len, seq_len)
            # We want attention from user profile (index 0) to passage (index i+1)
            passage_attention = attention_weights[:, 0, i+1]  # Shape: (batch_size,)
            
            # Calculate cosine similarity between reweighted user and passage
            passage_embedding = passage_embeddings[:, i, :]  # Shape: (batch_size, embedding_dim)
            
            # Normalize for cosine similarity
            user_norm = F.normalize(reweighted_user, p=2, dim=1)
            passage_norm = F.normalize(passage_embedding, p=2, dim=1)
            
            # Cosine similarity
            similarity = torch.sum(user_norm * passage_norm, dim=1)  # Shape: (batch_size,)
            
            # Combine attention weight with similarity
            attention_score = passage_attention * similarity
            attention_scores.append(attention_score)
        
        attention_scores = torch.stack(attention_scores, dim=1)  # Shape: (batch_size, num_passages)
        
        # If original scores provided, combine them
        if original_scores is not None:
            # Normalize both scores to [0, 1] range
            attention_scores_norm = (attention_scores - attention_scores.min(dim=1, keepdim=True)[0]) / \
                                  (attention_scores.max(dim=1, keepdim=True)[0] - attention_scores.min(dim=1, keepdim=True)[0] + 1e-8)
            original_scores_norm = (original_scores - original_scores.min(dim=1, keepdim=True)[0]) / \
                                 (original_scores.max(dim=1, keepdim=True)[0] - original_scores.min(dim=1, keepdim=True)[0] + 1e-8)
            
            # Combine scores (weighted average)
            final_scores = 0.6 * attention_scores_norm + 0.4 * original_scores_norm
        else:
            final_scores = attention_scores
        
        return final_scores, attention_weights

class PersonalizedSearchEngine:
    """
    Complete personalized search engine with FAISS retrieval and attention-based reranking.
    """
    
    def __init__(self, faiss_index_path="faiss_index/passage_index.faiss", 
                 user_profiles_path="user_profiles/user_profile_embeddings.npy",
                 user_mapping_path="user_profiles/user_mapping.json",
                 passage_embeddings_path="embeddings/passage_embeddings.npy"):
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        self.index.nprobe = 10
        
        # Load user profiles
        self.user_profiles = np.load(user_profiles_path)
        with open(user_mapping_path, 'r') as f:
            self.user_mapping = json.load(f)
        
        # Load passage embeddings for reranking
        self.passage_embeddings = np.load(passage_embeddings_path)
        
        # Initialize reranker
        self.reranker = PersonalizedReranker()
        
        print(f"Loaded {len(self.user_profiles)} user profiles")
        print(f"Loaded FAISS index with {self.index.ntotal} passages")
    
    def search_and_rerank(self, query_embedding, user_id, top_k=100, rerank_k=10):
        """
        Perform personalized search with reranking.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            user_id (str): User ID for personalization
            top_k (int): Number of passages to retrieve from FAISS
            rerank_k (int): Number of passages to return after reranking
        
        Returns:
            tuple: (reranked_indices, reranked_scores, attention_weights)
        """
        # Get user profile
        user_idx = int(user_id.split('_')[1])  # Extract user number from "user_X"
        user_profile = self.user_profiles[user_idx]
        
        # FAISS search
        query_tensor = query_embedding.astype(np.float32).reshape(1, -1)
        faiss_scores, faiss_indices = self.index.search(query_tensor, top_k)
        
        # Get passage embeddings for top-k results
        passage_embeddings = self.passage_embeddings[faiss_indices[0]]
        
        # Convert to tensors
        user_profile_tensor = torch.FloatTensor(user_profile).unsqueeze(0).to(self.reranker.device)
        passage_tensor = torch.FloatTensor(passage_embeddings).unsqueeze(0).to(self.reranker.device)
        faiss_scores_tensor = torch.FloatTensor(faiss_scores).to(self.reranker.device)
        
        # Rerank with attention
        with torch.no_grad():
            personalized_scores, attention_weights = self.reranker(
                user_profile_tensor, passage_tensor, faiss_scores_tensor
            )
        
        # Get top reranked results
        personalized_scores = personalized_scores.cpu().numpy().flatten()
        top_rerank_indices = np.argsort(personalized_scores)[::-1][:rerank_k]
        
        # Map back to original passage indices
        reranked_passage_indices = faiss_indices[0][top_rerank_indices]
        reranked_scores = personalized_scores[top_rerank_indices]
        
        return reranked_passage_indices, reranked_scores, attention_weights
    
    def batch_search_and_rerank(self, query_embeddings, user_ids, top_k=100, rerank_k=10):
        """
        Perform batch personalized search with reranking.
        
        Args:
            query_embeddings (np.ndarray): Query embeddings (num_queries, embedding_dim)
            user_ids (list): List of user IDs
            top_k (int): Number of passages to retrieve from FAISS
            rerank_k (int): Number of passages to return after reranking
        
        Returns:
            list: List of (reranked_indices, reranked_scores, attention_weights) tuples
        """
        results = []
        
        for i, (query_embedding, user_id) in enumerate(zip(query_embeddings, user_ids)):
            print(f"Processing query {i+1}/{len(query_embeddings)} for user {user_id}")
            
            result = self.search_and_rerank(query_embedding, user_id, top_k, rerank_k)
            results.append(result)
        
        return results

def test_personalized_search():
    """Test the personalized search engine with sample queries and users."""
    
    print("Testing personalized search engine...")
    
    # Load query embeddings
    query_embeddings = np.load("embeddings/query_embeddings.npy")
    
    # Initialize search engine
    search_engine = PersonalizedSearchEngine()
    
    # Test with sample queries and users
    sample_queries = query_embeddings[:5]
    sample_users = [f"user_{i}" for i in range(5)]
    
    print(f"Testing with {len(sample_queries)} queries and {len(sample_users)} users")
    
    results = search_engine.batch_search_and_rerank(
        sample_queries, sample_users, top_k=100, rerank_k=10
    )
    
    # Analyze results
    print(f"\nPersonalized Search Results:")
    for i, (reranked_indices, reranked_scores, attention_weights) in enumerate(results):
        print(f"\nQuery {i+1} for {sample_users[i]}:")
        print(f"  Top 5 reranked passages:")
        for j in range(min(5, len(reranked_indices))):
            print(f"    Passage {reranked_indices[j]}: Score {reranked_scores[j]:.4f}")
        
        # Analyze attention weights
        # attention_weights shape: (batch_size, seq_len, seq_len)
        # Get attention from user profile (index 0) to all passages
        user_attention = attention_weights[0, 0, 1:].cpu().numpy()  # Shape: (num_passages,)
        top_attention_indices = np.argsort(user_attention)[::-1][:5]
        print(f"  Top 5 attention-weighted passages:")
        for j, attention_idx in enumerate(top_attention_indices):
            # Map attention index to actual passage index
            if attention_idx < len(reranked_indices):
                passage_idx = reranked_indices[attention_idx]
                attention_value = user_attention[attention_idx]
                print(f"    Passage {passage_idx}: Attention {attention_value:.4f}")
            else:
                print(f"    Attention index {attention_idx} out of bounds")
    
    print(f"\nPersonalized search test completed successfully!")
    return results

def compare_with_baseline():
    """Compare personalized search with baseline FAISS search."""
    
    print("Comparing personalized search with baseline...")
    
    # Load query embeddings
    query_embeddings = np.load("embeddings/query_embeddings.npy")
    
    # Initialize search engine
    search_engine = PersonalizedSearchEngine()
    
    # Test with a few queries
    test_queries = query_embeddings[:3]
    test_users = ["user_0", "user_1", "user_2"]
    
    print(f"Comparing baseline vs personalized for {len(test_queries)} queries")
    
    for i, (query_embedding, user_id) in enumerate(zip(test_queries, test_users)):
        print(f"\nQuery {i+1} for {user_id}:")
        
        # Baseline FAISS search
        query_tensor = query_embedding.astype(np.float32).reshape(1, -1)
        faiss_scores, faiss_indices = search_engine.index.search(query_tensor, 10)
        
        print(f"  Baseline top 5:")
        for j in range(5):
            print(f"    Passage {faiss_indices[0][j]}: Score {faiss_scores[0][j]:.4f}")
        
        # Personalized search
        reranked_indices, reranked_scores, _ = search_engine.search_and_rerank(
            query_embedding, user_id, top_k=100, rerank_k=10
        )
        
        print(f"  Personalized top 5:")
        for j in range(5):
            print(f"    Passage {reranked_indices[j]}: Score {reranked_scores[j]:.4f}")
        
        # Calculate overlap
        baseline_set = set(faiss_indices[0][:10])
        personalized_set = set(reranked_indices[:10])
        overlap = len(baseline_set.intersection(personalized_set))
        overlap_ratio = overlap / 10
        
        print(f"  Overlap: {overlap}/10 ({overlap_ratio:.1%})")
    
    print(f"\nBaseline comparison completed!")

if __name__ == "__main__":
    # Test personalized search
    test_personalized_search()
    
    # Compare with baseline
    compare_with_baseline() 