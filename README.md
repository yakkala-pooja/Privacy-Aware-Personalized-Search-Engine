# Privacy-Aware Personalized Search Engine

This project implements a privacy-aware personalized search engine using the MS MARCO passage ranking dataset.

## Dataset Loading and Preprocessing

The project includes a script to load and preprocess the MS MARCO passage ranking dataset:

### Features:
- **Dataset**: MS MARCO v2.1 passage ranking dataset
- **Size**: 10K query-passage pairs (99,829 total samples after preprocessing)
- **Text Preprocessing**:
  - HTML tag removal
  - Lowercase conversion
  - Stopword removal
  - Whitespace normalization
  - Special character removal

### Dataset Structure:
- `query_id`: Unique identifier for each query-passage pair
- `query`: Preprocessed query text
- `passage`: Preprocessed passage text
- `relevance_score`: Binary relevance score (0 or 1)

### Statistics:
- **Total samples**: 99,829
- **Relevant samples**: 5,798 (5.8%)
- **Irrelevant samples**: 94,031 (94.2%)
- **Average query length**: 3.72 words
- **Average passage length**: 31.42 words

### Files:
- `load_marco_dataset.py`: Main script for loading and preprocessing
- `marco_preprocessed.csv`: Preprocessed dataset
- `verify_dataset.py`: Dataset verification script
- `requirements.txt`: Required dependencies

### Usage:
```bash
# Install dependencies
pip install -r requirements.txt

# Load and preprocess dataset
python load_marco_dataset.py

# Verify dataset quality
python verify_dataset.py
```

The preprocessed dataset is ready for downstream processing in privacy-aware search applications.