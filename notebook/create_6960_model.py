import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

def create_6960_model():
    print("Creating model for 6960 features...")
    
    X_train = np.load('../data/processed/X_train.npy')
    y_train = np.load('../data/processed/y_train.npy')
    
    print(f"Training data: {X_train.shape}")
    
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        max_depth=15
    )
    
    print("Training model with 6960 features...")
    model.fit(X_train, y_train)
    
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/sound_classifier.pkl'
    joblib.dump(model, model_path)
    
    train_score = model.score(X_train, y_train)
    print(f" Model created and saved to {model_path}")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Model expects {X_train.shape[1]} features")
    
    return True

if __name__ == "__main__":
    create_6960_model()