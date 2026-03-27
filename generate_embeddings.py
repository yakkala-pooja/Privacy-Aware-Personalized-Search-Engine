import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

class DistilBERTEmbedder:
    """
    A class to generate DistilBERT embeddings for queries and passages.
    Optimized for CPU usage to simulate on-device privacy.
    """
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=512, device=None):
        """
        Initialize the DistilBERT embedder.
        
        Args:
            model_name (str): HuggingFace model name
            max_length (int): Maximum sequence length for tokenization
            device (str): Device to use ('cpu' or 'cuda'). Defaults to 'cpu' for privacy
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Force CPU usage for privacy simulation
        self.device = torch.device('cpu') if device is None else torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts, batch_size=4, pooling_strategy='mean'):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            pooling_strategy (str): Pooling strategy ('mean', 'cls', 'max')
        
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        embeddings = []
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():  # No gradient computation for inference
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract hidden states
                hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                
                # Apply pooling strategy
                if pooling_strategy == 'cls':
                    # Use [CLS] token embedding
                    batch_embeddings = hidden_states[:, 0, :]
                elif pooling_strategy == 'mean':
                    # Mean pooling over sequence length (excluding padding)
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    masked_hidden_states = hidden_states * attention_mask_expanded
                    sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
                    seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
                    batch_embeddings = sum_hidden_states / seq_lengths
                elif pooling_strategy == 'max':
                    # Max pooling over sequence length
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    masked_hidden_states = hidden_states * attention_mask_expanded
                    batch_embeddings = torch.max(masked_hidden_states, dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
            
            # Convert to numpy and add to list
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            embeddings.append(batch_embeddings_np)
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def generate_query_embeddings(self, queries, batch_size=8):
        """
        Generate embeddings for queries.
        
        Args:
            queries (list): List of query strings
            batch_size (int): Batch size for processing
        
        Returns:
            np.ndarray: Query embeddings
        """
        print(f"Generating embeddings for {len(queries)} queries...")
        return self.generate_embeddings(queries, batch_size=batch_size, pooling_strategy='mean')
    
    def generate_passage_embeddings(self, passages, batch_size=2):
        """
        Generate embeddings for passages.
        
        Args:
            passages (list): List of passage strings
            batch_size (int): Batch size for processing (smaller for passages due to length)
        
        Returns:
            np.ndarray: Passage embeddings
        """
        print(f"Generating embeddings for {len(passages)} passages...")
        return self.generate_embeddings(passages, batch_size=batch_size, pooling_strategy='mean')

def load_and_embed_dataset(csv_file="marco_preprocessed.csv", 
                          embedder=None,
                          save_embeddings=True,
                          output_dir="embeddings"):
    """
    Load the preprocessed dataset and generate embeddings for queries and passages.
    
    Args:
        csv_file (str): Path to the preprocessed CSV file
        embedder (DistilBERTEmbedder): Pre-initialized embedder
        save_embeddings (bool): Whether to save embeddings to files
        output_dir (str): Directory to save embeddings
    
    Returns:
        tuple: (query_embeddings, passage_embeddings, df)
    """
    import os
    
    # Load dataset
    print(f"Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    
    # Initialize embedder if not provided
    if embedder is None:
        embedder = DistilBERTEmbedder()
    
    # Extract unique queries and passages
    unique_queries = df['query'].unique()
    unique_passages = df['passage'].unique()
    
    print(f"Unique queries: {len(unique_queries)}")
    print(f"Unique passages: {len(unique_passages)}")
    
    # Generate embeddings
    start_time = time.time()
    query_embeddings = embedder.generate_query_embeddings(unique_queries.tolist())
    query_time = time.time() - start_time
    print(f"Query embeddings generated in {query_time:.2f} seconds")
    
    start_time = time.time()
    passage_embeddings = embedder.generate_passage_embeddings(unique_passages.tolist())
    passage_time = time.time() - start_time
    print(f"Passage embeddings generated in {passage_time:.2f} seconds")
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Passage embeddings shape: {passage_embeddings.shape}")
    
    # Create mapping dictionaries for efficient lookup
    query_to_idx = {query: idx for idx, query in enumerate(unique_queries)}
    passage_to_idx = {passage: idx for idx, passage in enumerate(unique_passages)}
    
    # Add embedding indices to dataframe
    df['query_embedding_idx'] = df['query'].map(query_to_idx)
    df['passage_embedding_idx'] = df['passage'].map(passage_to_idx)
    
    # Save embeddings if requested
    if save_embeddings:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings as numpy arrays
        np.save(os.path.join(output_dir, 'query_embeddings.npy'), query_embeddings)
        np.save(os.path.join(output_dir, 'passage_embeddings.npy'), passage_embeddings)
        
        # Save mapping information
        mapping_df = pd.DataFrame({
            'query': unique_queries,
            'query_embedding_idx': range(len(unique_queries))
        })
        mapping_df.to_csv(os.path.join(output_dir, 'query_mapping.csv'), index=False)
        
        mapping_df = pd.DataFrame({
            'passage': unique_passages,
            'passage_embedding_idx': range(len(unique_passages))
        })
        mapping_df.to_csv(os.path.join(output_dir, 'passage_mapping.csv'), index=False)
        
        # Save updated dataset with embedding indices
        df.to_csv(os.path.join(output_dir, 'dataset_with_embeddings.csv'), index=False)
        
        print(f"Embeddings saved to {output_dir}/")
    
    return query_embeddings, passage_embeddings, df

def compute_similarity_scores(query_embeddings, passage_embeddings, query_idx, passage_indices):
    """
    Compute cosine similarity scores between a query and multiple passages.
    
    Args:
        query_embeddings (np.ndarray): All query embeddings
        passage_embeddings (np.ndarray): All passage embeddings
        query_idx (int): Index of the query
        passage_indices (list): Indices of passages to compare
    
    Returns:
        np.ndarray: Similarity scores
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_embedding = query_embeddings[query_idx].reshape(1, -1)
    passage_embeddings_subset = passage_embeddings[passage_indices]
    
    similarities = cosine_similarity(query_embedding, passage_embeddings_subset)[0]
    return similarities

if __name__ == "__main__":
    # Load and embed the dataset
    print("Starting embedding generation for entire dataset...")
    query_embeddings, passage_embeddings, df = load_and_embed_dataset()
    
    print("\nEmbedding generation complete!")
    print(f"Query embeddings: {query_embeddings.shape}")
    print(f"Passage embeddings: {passage_embeddings.shape}")
    print(f"Dataset with embedding indices: {df.shape}")
    
    # Example: Compute similarity for first query
    if len(df) > 0:
        first_query_idx = df.iloc[0]['query_embedding_idx']
        first_passage_idx = df.iloc[0]['passage_embedding_idx']
        
        # Get similar passages for the first query
        similarities = compute_similarity_scores(
            query_embeddings, 
            passage_embeddings, 
            first_query_idx, 
            [first_passage_idx]
        )
        
        print(f"\nExample similarity score for first query-passage pair: {similarities[0]:.4f}") 