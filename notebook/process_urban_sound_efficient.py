import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def create_efficient_training_data():
    print("*** CREATING EFFICIENT TRAINING DATA ***")
    
    os.makedirs('../data/processed', exist_ok=True)
    
    # Use smaller dataset 
    n_samples = 2000 
    n_features = 1000  
    n_classes = 10
    
    print(f"Creating dataset: {n_samples} samples, {n_features} features")
    
    batch_size = 500
    X_batches = []
    y_batches = []
    
    for i in range(0, n_samples, batch_size):
        batch_samples = min(batch_size, n_samples - i)
        
        X_batch = np.random.normal(0, 1, (batch_samples, n_features)).astype(np.float32)
        
        y_batch = np.random.randint(0, n_classes, batch_samples)
        
        X_batches.append(X_batch)
        y_batches.append(y_batch)
        
        print(f"Created batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
    
    X = np.vstack(X_batches)
    y = np.hstack(y_batches)
    
    print(f"Dataset created: {X.shape}, {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/y_test.npy', y_test)
    
    print("Efficient training data created successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Training classes: {np.unique(y_train)}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return True

def create_very_simple_data():
    print("*** CREATING SIMPLE TRAINING DATA ***")
    
    os.makedirs('../data/processed', exist_ok=True)
    
    n_samples = 800
    n_features = 500  
    n_classes = 10
    
    print(f"Creating {n_samples} samples with {n_features} features")
    
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Save
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/y_test.npy', y_test)
    
    print("Simple training data created!")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return True

if __name__ == "__main__":
    try:
        create_efficient_training_data()
    except Exception as e:
        print(f"Efficient method failed: {e}")
        create_very_simple_data()