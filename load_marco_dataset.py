import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Clean and preprocess text:
    - Remove HTML tags
    - Convert to lowercase
    - Remove stopwords
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Join tokens back into text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def load_marco_dataset(num_samples=10000):
    """
    Load MS MARCO passage ranking dataset and preprocess it.
    
    Args:
        num_samples (int): Number of samples to load (default: 10000)
    
    Returns:
        pd.DataFrame: Preprocessed dataset with columns: query_id, query, passage, relevance_score
    """
    print("Loading MS MARCO passage ranking dataset...")
    
    try:
        # Try to load a smaller subset first
        print("Attempting to load validation split for faster processing...")
        dataset = load_dataset("ms_marco", "v2.1", split="validation")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        print(f"Loaded {len(df)} samples from MS MARCO validation set")
        
        # If we have enough samples, use them; otherwise try train split
        if len(df) < num_samples:
            print("Validation set too small, loading train split...")
            dataset = load_dataset("ms_marco", "v2.1", split="train")
            df = pd.DataFrame(dataset)
            print(f"Loaded {len(df)} samples from MS MARCO train set")
        
        # Sample the requested number of rows
        if len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        
        print(f"Using {len(df)} samples from MS MARCO dataset")
        
    except Exception as e:
        print(f"Error loading MS MARCO dataset: {e}")
        print("Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset(num_samples)
    
    # Select and rename relevant columns
    print("Available columns:", df.columns.tolist())
    
    # MS MARCO has passages as a dictionary with parallel lists
    # We need to extract passages and create query-passage pairs
    print("Processing passages from MS MARCO dataset...")
    
    processed_data = []
    
    for idx, row in df.iterrows():
        query = row['query']
        query_id = row.get('query_id', f"q_{idx}")
        
        # Skip if query is empty or null
        if not query or pd.isna(query) or str(query).strip() == '':
            continue
        
        # Extract passages from the passages dictionary
        if 'passages' in row and isinstance(row['passages'], dict):
            passages_dict = row['passages']
            
            # Check if we have the expected structure
            if 'passage_text' in passages_dict and 'is_selected' in passages_dict:
                passage_texts = passages_dict['passage_text']
                is_selected_list = passages_dict['is_selected']
                
                # Ensure both lists have the same length
                if len(passage_texts) == len(is_selected_list):
                    for passage_idx, (passage_text, is_selected) in enumerate(zip(passage_texts, is_selected_list)):
                        # Only include passages with actual text
                        if passage_text and not pd.isna(passage_text) and str(passage_text).strip():
                            processed_data.append({
                                'query_id': f"{query_id}_{passage_idx}",
                                'query': query,
                                'passage': passage_text,
                                'relevance_score': is_selected
                            })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Clean the text data
    print("Preprocessing text data...")
    df['query'] = df['query'].apply(clean_text)
    df['passage'] = df['passage'].apply(clean_text)
    
    # Remove rows with empty text after cleaning
    df = df[df['query'].str.strip() != '']
    df = df[df['passage'].str.strip() != '']
    
    # Convert relevance_score to binary (0 or 1)
    df['relevance_score'] = (df['relevance_score'] > 0).astype(int)
    
    # Select only the required columns
    required_columns = ['query_id', 'query', 'passage', 'relevance_score']
    df = df[required_columns]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Preprocessing complete. Final dataset shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
    
    return df

def create_synthetic_dataset(num_samples=10000):
    """
    Create a synthetic dataset for demonstration purposes when MS MARCO is not available.
    """
    print("Creating synthetic query-passage dataset...")
    
    # Sample queries
    sample_queries = [
        "what is machine learning",
        "how to cook pasta",
        "best programming languages",
        "weather forecast today",
        "python tutorial for beginners",
        "healthy breakfast recipes",
        "how to train a dog",
        "best movies 2023",
        "how to lose weight",
        "javascript vs python"
    ]
    
    # Sample passages
    sample_passages = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "To cook pasta, bring a large pot of salted water to boil, add pasta, and cook according to package directions until al dente.",
        "Popular programming languages include Python, JavaScript, Java, C++, and Rust, each with their own strengths and use cases.",
        "Weather forecasting uses computer models and atmospheric data to predict future weather conditions and patterns.",
        "Python is a high-level programming language known for its simplicity and readability, making it perfect for beginners.",
        "A healthy breakfast might include oatmeal with berries, Greek yogurt with nuts, or whole grain toast with avocado.",
        "Training a dog requires consistency, positive reinforcement, and patience. Start with basic commands like sit and stay.",
        "The best movies of 2023 include Oppenheimer, Barbie, Killers of the Flower Moon, and Poor Things.",
        "To lose weight, focus on a balanced diet, regular exercise, adequate sleep, and stress management.",
        "JavaScript is primarily used for web development while Python is more versatile for data science, AI, and general programming."
    ]
    
    # Create synthetic data
    data = []
    for i in range(num_samples):
        query_idx = i % len(sample_queries)
        passage_idx = i % len(sample_passages)
        
        # Create some relevance variation
        relevance = 1 if (i % 3 == 0) else 0  # 33% relevant
        
        data.append({
            'query_id': f"q_{i}",
            'query': sample_queries[query_idx],
            'passage': sample_passages[passage_idx],
            'relevance_score': relevance
        })
    
    df = pd.DataFrame(data)
    
    # Clean the text data
    print("Preprocessing synthetic text data...")
    df['query'] = df['query'].apply(clean_text)
    df['passage'] = df['passage'].apply(clean_text)
    
    print(f"Synthetic dataset created with shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Load and preprocess the dataset
    df = load_marco_dataset(num_samples=10000)
    
    # Save the preprocessed dataset
    output_file = "marco_preprocessed.csv"
    df.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Relevant samples (relevance_score=1): {df['relevance_score'].sum()}")
    print(f"Irrelevant samples (relevance_score=0): {(df['relevance_score']==0).sum()}")
    print(f"Average query length: {df['query'].str.split().str.len().mean():.2f} words")
    print(f"Average passage length: {df['passage'].str.split().str.len().mean():.2f} words") 