from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path BEFORE importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Now import from src
from src.preprocessing import AudioPreprocessor, CLASS_NAMES

app = Flask(__name__)
CORS(app)

# Paths
MODEL_PATH = './models/sound_classifier.pkl'
UPLOAD_DIR = './data/upload'

# Global variables
model = None
preprocessor = None

def load_models():
    """Load ML models"""
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = AudioPreprocessor()
        print("Model loaded successfully from", MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        model = None
        preprocessor = None

# Load models on startup
load_models()

@app.route('/')
def home():
    return jsonify({
        'message': 'Sound Classification API',
        'status': 'running',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'classes_available': CLASS_NAMES,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sound class for single audio file"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded properly'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.flac']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'Only WAV, MP3, and FLAC files are supported'}), 400

        # Save uploaded file temporarily
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)
        file.save(file_path)

        # Extract features and predict
        features = preprocessor.extract_features(file_path)
        if features is None:
            os.remove(file_path)
            return jsonify({'error': 'Could not process audio file'}), 400

        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'class': CLASS_NAMES[i],
                'confidence': float(probabilities[i])
            }
            for i in top_indices
        ]

        # Clean up
        try:
            os.remove(file_path)
        except:
            pass

        return jsonify({
            'filename': file.filename,
            'predicted_class': CLASS_NAMES[prediction],
            'confidence': float(probabilities[prediction]),
            'all_predictions': top_predictions,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sound classes for multiple audio files"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded properly'}), 500

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files selected'}), 400

        results = []
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        for file in files:
            allowed_extensions = ['.wav', '.mp3', '.flac']
            if file.filename == '' or not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                continue

            temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            file_path = os.path.join(UPLOAD_DIR, temp_filename)
            file.save(file_path)

            features = preprocessor.extract_features(file_path)
            if features is not None:
                features = features.reshape(1, -1)
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                results.append({
                    'filename': file.filename,
                    'predicted_class': CLASS_NAMES[prediction],
                    'confidence': float(probabilities[prediction])
                })

            try:
                os.remove(file_path)
            except:
                pass

        return jsonify({
            'predictions': results,
            'total_files_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    return jsonify({
        'model_type': type(model).__name__,
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'feature_size': preprocessor.get_feature_size(),
        'sample_rate': preprocessor.sr,
        'n_mfcc': preprocessor.n_mfcc,
        'max_length': preprocessor.max_len
    })

@app.route('/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining with new data"""
    try:
        from src.model import ModelRetrainer
        
        retrainer = ModelRetrainer(MODEL_PATH)
        results = retrainer.retrain(save_path=MODEL_PATH)
        
        # Reload the model
        load_models()
        
        return jsonify({
            'message': 'Retraining completed successfully',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Retraining failed'
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Sound Classification API Starting...")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)