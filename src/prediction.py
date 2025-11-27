import numpy as np
import joblib
import os
from .preprocessing import AudioPreprocessor, CLASS_NAMES

class SoundPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.preprocessor = AudioPreprocessor()
        self.class_names = CLASS_NAMES
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, audio_file_path):
        if self.model is None:
            return {
                'error': 'Model not loaded. Please load a trained model first.',
                'prediction': None,
                'confidence': None,
                'class_name': None
            }
        
        try:
            features = self.preprocessor.extract_features(audio_file_path)
            
            if features is None:
                return {
                    'error': 'Failed to extract features from audio file',
                    'prediction': None,
                    'confidence': None,
                    'class_name': None
                }
            
            features = features.reshape(1, -1)
            
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
            
            class_name = self.class_names[prediction] if prediction < len(self.class_names) else f"Class_{prediction}"
            
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = [
                {
                    'class_name': self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                    'confidence': float(probabilities[i])
                }
                for i in top_indices
            ]
            
            return {
                'error': None,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'class_name': class_name,
                'all_predictions': top_predictions,
                'probabilities': {
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(prob)
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'confidence': None,
                'class_name': None
            }
    
    def predict_batch(self, audio_file_paths):
        results = []
        for file_path in audio_file_paths:
            result = self.predict(file_path)
            result['file_path'] = file_path
            results.append(result)
        return results
    
    def get_model_info(self):
        if self.model is None:
            return {
                'model_type': 'Not loaded',
                'num_classes': len(self.class_names),
                'feature_size': self.preprocessor.get_feature_size(),
                'status': 'Not loaded',
                'model_path': self.model_path
            }
        
        model_type = type(self.model).__name__
        feature_size = self.preprocessor.get_feature_size()
        
        return {
            'model_type': model_type,
            'num_classes': len(self.class_names),
            'feature_size': feature_size,
            'status': 'Loaded',
            'model_path': self.model_path,
            'class_names': self.class_names
        }


def predict_sound(model_path, audio_file_path):
    predictor = SoundPredictor(model_path)
    return predictor.predict(audio_file_path)


if __name__ == "__main__":
    predictor = SoundPredictor()
    print("SoundPredictor class initialized successfully")
    print(f"Available class names: {CLASS_NAMES}")