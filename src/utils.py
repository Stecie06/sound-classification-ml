import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json


def plot_confusion_matrix(cm, class_names, figsize=(10, 8), save_path=None):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return plt.gcf()


def plot_class_distribution(labels, class_names, figsize=(12, 5), save_path=None):
    class_counts = [np.sum(labels == i) for i in range(len(class_names))]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(class_names, class_counts, color='lightblue', edgecolor='black')
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Sound Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    return plt.gcf()


def plot_feature_importance(importances, top_n=20, figsize=(12, 6), save_path=None):
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=figsize)
    plt.bar(range(top_n), top_importances, color='orange', alpha=0.7)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(range(top_n), [f'F{idx}' for idx in top_indices], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    return plt.gcf()


def save_metrics_to_json(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    metrics_clean = convert_types(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=4)
    
    print(f"Metrics saved to: {filepath}")


def load_metrics_from_json(filepath):
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def create_prediction_summary(predictions):
    if not predictions:
        return None
    
    confidences = [p['confidence'] for p in predictions]
    predicted_classes = [p['predicted_class'] for p in predictions]
    
    # Count predictions per class
    class_counts = {}
    for cls in predicted_classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    summary = {
        'total_predictions': len(predictions),
        'average_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'std_confidence': np.std(confidences),
        'class_distribution': class_counts,
        'high_confidence_count': sum(1 for c in confidences if c > 0.8),
        'low_confidence_count': sum(1 for c in confidences if c < 0.5)
    }
    
    return summary


def format_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def get_audio_files_from_dir(directory, extensions=['.wav', '.mp3', '.flac']):
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def calculate_model_metrics(y_true, y_pred, class_names):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    }
    
    return metrics

def load_model(model_path):
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_uploaded_file(uploaded_file, save_dir="temp_uploads"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def allowed_file(filename, allowed_extensions=None):
    if allowed_extensions is None:
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext.replace('.', '') for ext in allowed_extensions]

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

# Add CLASS_NAMES constant to utils.py as well (or keep it in preprocessing)
CLASS_NAMES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", 
               "drilling", "engine_idling", "gun_shot", "jackhammer", 
               "siren", "street_music"]