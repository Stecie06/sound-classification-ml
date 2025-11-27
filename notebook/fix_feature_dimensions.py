
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_compatible_training_data():
    
    print("Creating compatible training data with 6960 features...")
    
    os.makedirs('../data/processed', exist_ok=True)

    n_features = 40 * 174  
    n_samples = 1000
    n_classes = 10
    
    print(f"Creating {n_samples} samples with {n_features} features")

    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the data
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/y_test.npy', y_test)
    
    print("Compatible training data created successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Features match MFCC dimensions: {X_train.shape[1] == 6960}")
    
    return True

if __name__ == "__main__":
    create_compatible_training_data()