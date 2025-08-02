import pandas as pd
import numpy as np

# Load the preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv("marco_preprocessed.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Basic statistics
print(f"\nBasic Statistics:")
print(f"Total samples: {len(df)}")
print(f"Relevant samples (relevance_score=1): {df['relevance_score'].sum()}")
print(f"Irrelevant samples (relevance_score=0): {(df['relevance_score']==0).sum()}")
print(f"Relevance ratio: {df['relevance_score'].mean():.3f}")

# Text length statistics
print(f"\nText Length Statistics:")
print(f"Average query length: {df['query'].str.split().str.len().mean():.2f} words")
print(f"Median query length: {df['query'].str.split().str.len().median():.2f} words")
print(f"Average passage length: {df['passage'].str.split().str.len().mean():.2f} words")
print(f"Median passage length: {df['passage'].str.split().str.len().median():.2f} words")

# Check for empty texts
empty_queries = df['query'].str.strip().eq('').sum()
empty_passages = df['passage'].str.strip().eq('').sum()
print(f"\nEmpty text counts:")
print(f"Empty queries: {empty_queries}")
print(f"Empty passages: {empty_passages}")

# Sample data
print(f"\nSample data (first 5 rows):")
print(df.head().to_string())

# Check unique query IDs
unique_queries = df['query_id'].str.split('_').str[0].nunique()
print(f"\nUnique queries: {unique_queries}")

# Check passages per query
passages_per_query = df['query_id'].str.split('_').str[0].value_counts()
print(f"Passages per query statistics:")
print(f"  Min: {passages_per_query.min()}")
print(f"  Max: {passages_per_query.max()}")
print(f"  Mean: {passages_per_query.mean():.2f}")
print(f"  Median: {passages_per_query.median():.2f}")

# Check for any data quality issues
print(f"\nData Quality Check:")
print(f"Null values in query: {df['query'].isnull().sum()}")
print(f"Null values in passage: {df['passage'].isnull().sum()}")
print(f"Null values in relevance_score: {df['relevance_score'].isnull().sum()}")

# Check relevance score distribution
print(f"\nRelevance Score Distribution:")
relevance_counts = df['relevance_score'].value_counts().sort_index()
for score, count in relevance_counts.items():
    print(f"  Score {score}: {count} samples ({count/len(df)*100:.1f}%)")

print(f"\nDataset verification complete!") 