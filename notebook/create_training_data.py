import numpy as np
import os

def create_training_data():
    
    print("Creating training data for: ~/Documents/sound-classification-ml")
    
    os.makedirs('./data/processed', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    n_samples = 1000
    n_features = 40 * 174  
    
    # Create feature matrix 
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    
    # Create labels
    y = np.random.randint(0, 10, n_samples)
    
    # train/test split 
    split_idx = int(0.8 * n_samples)  
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Save the data
    np.save('./data/processed/X_train.npy', X_train)
    np.save('./data/processed/X_test.npy', X_test)
    np.save('./data/processed/y_train.npy', y_train)
    np.save('./data/processed/y_test.npy', y_test)
    
    print("Training data created successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}") 
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    required_files = [
        './data/processed/X_train.npy',
        './data/processed/X_test.npy',
        './data/processed/y_train.npy', 
        './data/processed/y_test.npy'
    ]
    
    existing_files = [f for f in required_files if os.path.exists(f)]
    print(f" Created {len(existing_files)}/{len(required_files)} required files")
    
    return True

if __name__ == "__main__":
    create_training_data()