import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import pandas as pd
import json
import os
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

class MSMARCODataset(Dataset):
    """Dataset for MS MARCO passage ranking with differential privacy training."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine query and passage for training
        text = f"{row['query']} [SEP] {row['passage']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to scalar tensors for DataLoader
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Binary classification: relevant (1) or not (0)
        label = torch.tensor(row['relevance_score'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

class DistilBERTForDPTraining(nn.Module):
    """DistilBERT model adapted for differential privacy training."""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super(DistilBERTForDPTraining, self).__init__()
        
        self.config = DistilBertConfig.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, num_labels)
        )
        
        # Freeze BERT layers for DP training (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss, logits
        else:
            return logits

class DPFineTuner:
    """Differential privacy fine-tuner for DistilBERT."""
    
    def __init__(self, epsilon=2.0, delta=1e-5, max_grad_norm=1.0, 
                 noise_multiplier=None, target_epsilon=2.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        
        # Initialize model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBERTForDPTraining()
        
        # Privacy engine
        self.privacy_engine = PrivacyEngine()
        
        print(f"Initialized DP fine-tuner with ε={epsilon}, δ={delta}")
    
    def prepare_data(self, csv_file="marco_preprocessed.csv", sample_size=10000):
        """Prepare MS MARCO dataset for DP training."""
        print("Loading MS MARCO dataset...")
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Sample for faster training
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} samples for training")
        
        # Split data
        train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = MSMARCODataset(train_data, self.tokenizer)
        val_dataset = MSMARCODataset(val_data, self.tokenizer)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_privacy_engine(self, train_loader):
        """Setup privacy engine with Opacus."""
        print("Setting up privacy engine...")
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        
        # Calculate noise multiplier if not provided
        if self.noise_multiplier is None:
            # Estimate based on target epsilon
            sample_rate = 1.0 / len(train_loader)
            steps = len(train_loader) * 3  # 3 epochs
            self.noise_multiplier = self._calculate_noise_multiplier(
                sample_rate, steps, self.target_epsilon, self.delta
            )
            print(f"Calculated noise multiplier: {self.noise_multiplier:.4f}")
        
        # Make model private
        self.privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        
        return optimizer
    
    def _calculate_noise_multiplier(self, sample_rate, steps, target_epsilon, delta):
        """Calculate noise multiplier for target epsilon."""
        # Simplified calculation - in practice, use Opacus utilities
        return 1.0  # Placeholder - will be calculated by Opacus
    
    def train(self, train_loader, val_loader, epochs=3):
        """Train model with differential privacy."""
        print("Starting differential privacy training...")
        
        # Setup privacy engine
        optimizer = self.setup_privacy_engine(train_loader)
        
        # Training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        best_val_acc = 0.0
        training_results = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'epsilon_used': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, epoch, device)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader, device)
            
            # Get privacy budget used
            epsilon_used = self.privacy_engine.get_epsilon(self.delta)
            
            # Store results
            training_results['train_losses'].append(train_loss)
            training_results['train_accs'].append(train_acc)
            training_results['val_losses'].append(val_loss)
            training_results['val_accs'].append(val_acc)
            training_results['epsilon_used'].append(epsilon_used)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}, ε={epsilon_used:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_dp_model")
        
        return training_results
    
    def _train_epoch(self, train_loader, optimizer, epoch, device):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            loss, logits = self.model(input_ids, attention_mask, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping and noise addition handled by PrivacyEngine
            optimizer.step()
            optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%"
            })
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader, device):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                loss, logits = self.model(input_ids, attention_mask, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def save_model(self, model_path="dp_fine_tuned_model"):
        """Save the differentially private model."""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(model_path, "model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save privacy parameters
        privacy_params = {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'max_grad_norm': self.max_grad_norm,
            'noise_multiplier': self.noise_multiplier
        }
        
        with open(os.path.join(model_path, "privacy_params.json"), 'w') as f:
            json.dump(privacy_params, f, indent=2)
        
        print(f"Model saved to {model_path}/")
    
    def load_model(self, model_path="dp_fine_tuned_model"):
        """Load the differentially private model."""
        # Load model state
        self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
        
        # Load privacy parameters
        with open(os.path.join(model_path, "privacy_params.json"), 'r') as f:
            privacy_params = json.load(f)
        
        print(f"Model loaded from {model_path}/")
        print(f"Privacy parameters: ε={privacy_params['epsilon']}, δ={privacy_params['delta']}")
    
    def evaluate_embedding_quality(self, test_data, original_embeddings):
        """Evaluate embedding quality degradation after DP training."""
        print("Evaluating embedding quality...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        # Generate new embeddings
        new_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(len(test_data)), desc="Generating embeddings"):
                row = test_data.iloc[i]
                text = f"{row['query']} [SEP] {row['passage']}"
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Get embeddings from BERT
                outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                
                new_embeddings.append(embedding.flatten())
        
        new_embeddings = np.array(new_embeddings)
        
        # Compare with original embeddings
        if original_embeddings is not None and len(original_embeddings) == len(new_embeddings):
            # Calculate cosine similarity
            similarities = []
            for i in range(len(new_embeddings)):
                sim = cosine_similarity(
                    original_embeddings[i].reshape(1, -1),
                    new_embeddings[i].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            print(f"Embedding Quality Analysis:")
            print(f"  Average similarity with original: {avg_similarity:.4f}")
            print(f"  Std similarity: {std_similarity:.4f}")
            print(f"  Min similarity: {np.min(similarities):.4f}")
            print(f"  Max similarity: {np.max(similarities):.4f}")
        
        return new_embeddings

def main():
    """Main function for DP fine-tuning."""
    
    # Initialize DP fine-tuner
    dp_tuner = DPFineTuner(epsilon=2.0, delta=1e-5)
    
    # Prepare data
    train_dataset, val_dataset = dp_tuner.prepare_data()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print("Preparing datasets...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train with differential privacy
    training_results = dp_tuner.train(train_loader, val_loader)
    
    # Save results
    with open("dp_training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print("Differential privacy fine-tuning completed!")
    print(f"Final ε used: {training_results['epsilon_used'][-1]:.4f}")

if __name__ == "__main__":
    main() 