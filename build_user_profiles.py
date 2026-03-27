import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class UserProfileTransformer(nn.Module):
    """
    Lightweight transformer encoder for building user profiles from clicked passages.
    Uses attention-weighted aggregation with learnable query context.
    """
    
    def __init__(self, embedding_dim=768, num_heads=8, num_layers=2, dropout=0.1):
        super(UserProfileTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Learnable query context vector
        self.query_context = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, clicked_embeddings, mask=None):
        """
        Forward pass for user profile generation.
        
        Args:
            clicked_embeddings (torch.Tensor): Shape (batch_size, num_clicks, embedding_dim)
            mask (torch.Tensor): Attention mask for padding
        
        Returns:
            torch.Tensor: User profile embedding (batch_size, embedding_dim)
        """
        batch_size, num_clicks, embedding_dim = clicked_embeddings.shape
        
        # Add learnable query context to the beginning
        query_context = self.query_context.expand(batch_size, 1, embedding_dim)
        sequence = torch.cat([query_context, clicked_embeddings], dim=1)
        
        # Create attention mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, num_clicks + 1, dtype=torch.bool)
        
        # Self-attention
        attended, _ = self.self_attention(
            sequence, sequence, sequence,
            key_padding_mask=~mask
        )
        
        # Residual connection and normalization
        attended = self.norm1(sequence + self.dropout(attended))
        
        # Feed-forward
        ff_output = self.feed_forward(attended)
        ff_output = self.norm2(attended + self.dropout(ff_output))
        
        # Take the query context position as the user profile
        user_profile = ff_output[:, 0, :]  # Shape: (batch_size, embedding_dim)
        
        # Final projection
        user_profile = self.output_projection(user_profile)
        
        return user_profile

class UserProfileBuilder:
    """
    Build user profiles from simulated historical clicks using transformer aggregation.
    """
    
    def __init__(self, embedding_dim=768, num_heads=8, num_layers=2):
        self.embedding_dim = embedding_dim
        self.transformer = UserProfileTransformer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        self.device = torch.device('cpu')  # Use CPU for privacy
        self.transformer.to(self.device)
        self.transformer.eval()
        
        # User profiles storage
        self.user_profiles = {}
        self.user_clicks = {}
    
    def simulate_user_clicks(self, num_users=100, min_clicks=20, max_clicks=100, 
                           passage_embeddings=None, dataset_df=None):
        """
        Simulate historical clicks for users.
        
        Args:
            num_users (int): Number of users to simulate
            min_clicks (int): Minimum clicks per user
            max_clicks (int): Maximum clicks per user
            passage_embeddings (np.ndarray): Passage embeddings
            dataset_df (pd.DataFrame): Dataset with passage information
        
        Returns:
            dict: User click histories
        """
        print(f"Simulating {num_users} users with {min_clicks}-{max_clicks} clicks each...")
        
        user_clicks = {}
        
        # Get relevant passages (those with relevance_score=1)
        relevant_passages = dataset_df[dataset_df['relevance_score'] == 1]
        relevant_indices = relevant_passages['passage_embedding_idx'].values
        
        # Also include some irrelevant passages for realism
        irrelevant_passages = dataset_df[dataset_df['relevance_score'] == 0]
        irrelevant_indices = irrelevant_passages['passage_embedding_idx'].values
        
        for user_id in range(num_users):
            # Determine number of clicks for this user
            num_clicks = random.randint(min_clicks, max_clicks)
            
            # Simulate click history with bias toward relevant passages
            click_history = []
            
            # 70% relevant passages, 30% irrelevant for realism
            num_relevant = int(num_clicks * 0.7)
            num_irrelevant = num_clicks - num_relevant
            
            # Sample relevant passages
            if len(relevant_indices) > 0:
                relevant_clicks = np.random.choice(
                    relevant_indices, 
                    size=min(num_relevant, len(relevant_indices)), 
                    replace=False
                )
                click_history.extend(relevant_clicks)
            
            # Sample irrelevant passages
            if len(irrelevant_indices) > 0:
                irrelevant_clicks = np.random.choice(
                    irrelevant_indices,
                    size=min(num_irrelevant, len(irrelevant_indices)),
                    replace=False
                )
                click_history.extend(irrelevant_clicks)
            
            # Shuffle the click history
            random.shuffle(click_history)
            
            user_clicks[f"user_{user_id}"] = click_history
        
        # Store in the class
        self.user_clicks = user_clicks
        
        print(f"Generated click histories for {len(user_clicks)} users")
        return user_clicks
    
    def build_user_profiles(self, user_clicks, passage_embeddings, max_sequence_length=50):
        """
        Build user profiles using transformer aggregation.
        
        Args:
            user_clicks (dict): User click histories
            passage_embeddings (np.ndarray): Passage embeddings
            max_sequence_length (int): Maximum sequence length for transformer
        
        Returns:
            dict: User profile embeddings
        """
        print("Building user profiles using transformer aggregation...")
        
        user_profiles = {}
        
        with torch.no_grad():
            for user_id, click_indices in tqdm(user_clicks.items(), desc="Building profiles"):
                # Get clicked passage embeddings
                clicked_embeddings = passage_embeddings[click_indices]
                
                # Truncate if too long
                if len(clicked_embeddings) > max_sequence_length:
                    clicked_embeddings = clicked_embeddings[:max_sequence_length]
                
                # Convert to tensor
                clicked_tensor = torch.FloatTensor(clicked_embeddings).unsqueeze(0).to(self.device)
                
                # Generate user profile
                user_profile = self.transformer(clicked_tensor)
                
                # Convert to numpy and store
                user_profiles[user_id] = user_profile.cpu().numpy().flatten()
        
        self.user_profiles = user_profiles
        print(f"Built profiles for {len(user_profiles)} users")
        
        return user_profiles
    
    def save_user_profiles(self, output_dir="user_profiles"):
        """Save user profiles to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy array
        user_ids = list(self.user_profiles.keys())
        profile_embeddings = np.array([self.user_profiles[uid] for uid in user_ids])
        
        np.save(os.path.join(output_dir, "user_profile_embeddings.npy"), profile_embeddings)
        
        # Save user ID mapping
        user_mapping = {i: user_id for i, user_id in enumerate(user_ids)}
        with open(os.path.join(output_dir, "user_mapping.json"), 'w') as f:
            json.dump(user_mapping, f, indent=2)
        
        # Save click histories (convert numpy int64 to regular int)
        serializable_clicks = {}
        for user_id, click_indices in self.user_clicks.items():
            serializable_clicks[user_id] = [int(idx) for idx in click_indices]
        
        with open(os.path.join(output_dir, "user_clicks.json"), 'w') as f:
            json.dump(serializable_clicks, f, indent=2)
        
        print(f"User profiles saved to {output_dir}/")
        print(f"Profile embeddings shape: {profile_embeddings.shape}")
        
        return profile_embeddings
    
    def load_user_profiles(self, output_dir="user_profiles"):
        """Load user profiles from disk."""
        profile_embeddings = np.load(os.path.join(output_dir, "user_profile_embeddings.npy"))
        
        with open(os.path.join(output_dir, "user_mapping.json"), 'r') as f:
            user_mapping = json.load(f)
        
        # Reconstruct user_profiles dict
        self.user_profiles = {user_mapping[str(i)]: profile_embeddings[i] 
                             for i in range(len(profile_embeddings))}
        
        return self.user_profiles

def analyze_user_profiles(user_profiles, query_embeddings=None):
    """Analyze user profile characteristics."""
    print("\nUser Profile Analysis:")
    
    # Convert to numpy array
    profiles_array = np.array(list(user_profiles.values()))
    
    print(f"Number of users: {len(user_profiles)}")
    print(f"Profile dimension: {profiles_array.shape[1]}")
    print(f"Profile statistics:")
    print(f"  Mean: {np.mean(profiles_array):.6f}")
    print(f"  Std: {np.std(profiles_array):.6f}")
    print(f"  Min: {np.min(profiles_array):.6f}")
    print(f"  Max: {np.max(profiles_array):.6f}")
    
    # Calculate profile similarities
    profile_similarities = cosine_similarity(profiles_array)
    print(f"Profile similarity statistics:")
    print(f"  Mean similarity: {np.mean(profile_similarities):.4f}")
    print(f"  Std similarity: {np.std(profile_similarities):.4f}")
    
    # If query embeddings provided, analyze query-user similarities
    if query_embeddings is not None:
        print(f"\nQuery-User Similarity Analysis:")
        sample_queries = query_embeddings[:10]
        query_user_similarities = cosine_similarity(sample_queries, profiles_array)
        
        print(f"  Mean query-user similarity: {np.mean(query_user_similarities):.4f}")
        print(f"  Std query-user similarity: {np.std(query_user_similarities):.4f}")

if __name__ == "__main__":
    # Load data
    print("Loading embeddings and dataset...")
    passage_embeddings = np.load("embeddings/passage_embeddings.npy")
    query_embeddings = np.load("embeddings/query_embeddings.npy")
    dataset_df = pd.read_csv("embeddings/dataset_with_embeddings.csv")
    
    print(f"Passage embeddings: {passage_embeddings.shape}")
    print(f"Query embeddings: {query_embeddings.shape}")
    print(f"Dataset: {dataset_df.shape}")
    
    # Initialize user profile builder
    profile_builder = UserProfileBuilder()
    
    # Simulate user clicks
    user_clicks = profile_builder.simulate_user_clicks(
        num_users=100,
        min_clicks=20,
        max_clicks=100,
        passage_embeddings=passage_embeddings,
        dataset_df=dataset_df
    )
    
    # Build user profiles
    user_profiles = profile_builder.build_user_profiles(
        user_clicks, 
        passage_embeddings
    )
    
    # Save user profiles
    profile_embeddings = profile_builder.save_user_profiles()
    
    # Analyze user profiles
    analyze_user_profiles(user_profiles, query_embeddings)
    
    print("\nUser profile generation completed successfully!") 