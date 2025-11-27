from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys
from datetime import datetime
import soundfile as sf

sys.path.append('./src')

app = Flask(__name__)
CORS(app)

MODEL_PATH = './models/sound_classifier.pkl'
PREPROCESSOR_PATH = './models/audio_preprocessor.pkl'

CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
    'siren', 'street_music'
]

model = None
preprocessor = None

def load_models():
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

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
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.wav'):
            return jsonify({'error': 'Only WAV files are supported'}), 400

        upload_dir = './data/upload'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        features = preprocessor.extract_features(file_path)
        if features is None:
            os.remove(file_path)
            return jsonify({'error': 'Could not process audio file'}), 400

        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'class': CLASS_NAMES[i],
                'confidence': float(probabilities[i])
            }
            for i in top_indices
        ]

        os.remove(file_path)

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
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files selected'}), 400

        results = []
        upload_dir = './data/upload'
        os.makedirs(upload_dir, exist_ok=True)

        for file in files:
            if file.filename == '' or not file.filename.lower().endswith('.wav'):
                continue

            file_path = os.path.join(upload_dir, file.filename)
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

            os.remove(file_path)

        return jsonify({
            'predictions': results,
            'total_files_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining with new data"""
    try:
        return jsonify({
            'message': 'Retraining triggered',
            'status': 'processing',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'feature_size': preprocessor.n_mfcc * preprocessor.max_len
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
