import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import time
from .preprocessing import AudioPreprocessor, CLASS_NAMES

class SoundClassifier:
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42,
                verbose=False
            )
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                random_state=42,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training {self.model_type} model...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
        
        return train_score, test_score, X_train, X_test, y_train, y_test
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=CLASS_NAMES,
                output_dict=True
            )
        }
        
        return metrics
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def train_model(X, y, model_type='random_forest', test_size=0.2):
    classifier = SoundClassifier(model_type=model_type)
    train_score, test_score, _, _, _, _ = classifier.train(X, y, test_size)
    return classifier, train_score, test_score


class ModelRetrainer:
    
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path
        self.model = None
        self.preprocessor = AudioPreprocessor()
    
    def load_base_model(self):
        if os.path.exists(self.base_model_path):
            self.model = joblib.load(self.base_model_path)
            print(f"Base model loaded from {self.base_model_path}")
            return True
        print(f"Base model not found at {self.base_model_path}")
        return False
    
    def load_original_data(self, data_dir='./data/processed'):
        try:
            X_path = os.path.join(data_dir, 'X_train.npy')
            y_path = os.path.join(data_dir, 'y_train.npy')
            
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                print(f"Original data not found in {data_dir}")
                return None, None
            
            X_original = np.load(X_path)
            y_original = np.load(y_path)
            print(f"Loaded {len(X_original)} original training samples")
            return X_original, y_original
        except Exception as e:
            print(f"Error loading original data: {e}")
            return None, None
    
    def process_new_audio_files(self, upload_dir='./data/upload'):
        if not os.path.exists(upload_dir):
            print(f"Upload directory not found: {upload_dir}")
            return np.array([]), np.array([])
        
        features = []
        labels = []
        file_count = 0
        
        print(f"Processing new audio files from {upload_dir}...")
        
        for class_name in os.listdir(upload_dir):
            class_dir = os.path.join(upload_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            if class_name not in CLASS_NAMES:
                print(f"Warning: Unknown class '{class_name}', skipping...")
                continue
                
            class_index = CLASS_NAMES.index(class_name)
            
            for audio_file in os.listdir(class_dir):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(class_dir, audio_file)
                    try:
                        feature = self.preprocessor.extract_features(audio_path)
                        if feature is not None:
                            features.append(feature)
                            labels.append(class_index)
                            file_count += 1
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
        
        print(f"Processed {file_count} new audio files")
        return np.array(features), np.array(labels)
    
    def retrain(self, save_path=None):
        # Load base model
        if self.model is None:
            if not self.load_base_model():
                raise ValueError("No base model found. Cannot proceed with retraining.")

        X_original, y_original = self.load_original_data()
        if X_original is None:
            raise ValueError("Original training data not found. Please run the notebook first to generate training data.")

        X_new, y_new = self.process_new_audio_files()
        
        if len(X_new) == 0:
            raise ValueError("No new audio files found for retraining. Please add audio files to ./data/upload/")
       
        X_combined = np.vstack([X_original, X_new])
        y_combined = np.hstack([y_original, y_new])
        
        print("\n" + "="*50)
        print("RETRAINING SUMMARY")
        print("="*50)
        print(f"Original data: {len(X_original)} samples")
        print(f"New data: {len(X_new)} samples")
        print(f"Combined data: {len(X_combined)} samples")
        print("="*50 + "\n")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        print("Retraining model...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"\nRetraining completed in {training_time:.2f} seconds")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"\nRetrained model saved to {save_path}")
        
        results = {
            'training_time': training_time,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'original_samples': int(len(X_original)),
            'new_samples': int(len(X_new)),
            'total_samples': int(len(X_combined))
        }
        
        return results


if __name__ == "__main__":
    print("Sound Classification Model Module")
    print(f"Available classes: {CLASS_NAMES}")