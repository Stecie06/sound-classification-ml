
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
from datetime import datetime
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

try:
    from src.preprocessing import AudioPreprocessor
    from src.prediction import SoundPredictor
    from src.model import ModelRetrainer
    from src.utils import (
        plot_confusion_matrix, plot_class_distribution, 
        plot_feature_importance, create_prediction_summary
    )
except ImportError as e:
    st.error(f"Error importing modules: {e}")

CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
    'siren', 'street_music'
]

MODEL_PATH = './models/sound_classifier.pkl'
PREPROCESSOR_PATH = './models/audio_preprocessor.pkl'
UPLOAD_DIR = './data/upload'

st.set_page_config(
    page_title="Sound Classification ML Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<link href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css' rel='stylesheet'>", unsafe_allow_html=True)


def safe_dataframe_display(df, width='stretch'):
    if df is None or df.empty:
        st.info("No data to display")
        return
        
    df_display = df.copy()
    
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            try:
                df_display[col] = df_display[col].astype(str)
            except Exception:
                pass
    
    # Display the fixed dataframe
    st.dataframe(df_display, width=width)

def safe_table_display(data):
    if isinstance(data, pd.DataFrame):
        st.table(data.astype(str))
    else:
        st.write(data)

@st.cache_resource
def load_model_and_preprocessor():
    try:
        predictor = SoundPredictor(MODEL_PATH)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_model_uptime():
    if os.path.exists(MODEL_PATH):
        model_time = os.path.getmtime(MODEL_PATH)
        uptime_seconds = time.time() - model_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    return "N/A"


def check_training_data():
    required_files = [
        './data/processed/X_train.npy',
        './data/processed/y_train.npy', 
        './data/processed/X_test.npy',
        './data/processed/y_test.npy'
    ]
    
    existing_files = [f for f in required_files if os.path.exists(f)]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    return len(existing_files) == len(required_files), existing_files, missing_files


def create_missing_training_data():
    try:
        script_path = os.path.join(PROJECT_ROOT, 'notebook', 'create_training_data.py')
        
        if os.path.exists(script_path):
            import subprocess
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                return True, "Training data created successfully!"
            else:
                return False, f"Error: {result.stderr}"
        else:
            return False, "Training data script not found"
            
    except Exception as e:
        return False, f"Error creating training data: {e}"


def main():

    st.markdown('<h1 class="main-header"><i class="fa fa-volume-up" aria-hidden="true"></i> Sound Classification ML Pipeline</h1>', 
                unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Predict", "Visualizations", "Upload & Retrain", "Monitoring"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About**: This ML pipeline classifies environmental sounds into 10 categories "
        "using MFCC features and Random Forest."
    )
    
    # Load predictor
    predictor = load_model_and_preprocessor()
    
    if predictor is None:
        st.error("Failed to load model. Please ensure model files exist in ./models/")
        return
    
    # Page routing
    if page == "Home":
        show_home_page(predictor)
    elif page == "Predict":
        show_prediction_page(predictor)
    elif page == "Visualizations":
        show_visualization_page(predictor)
    elif page == "Upload & Retrain":
        show_upload_retrain_page(predictor)
    elif page == "Monitoring":
        show_monitoring_page(predictor)


def show_home_page(predictor):

    st.header("Welcome to Sound Classification ML Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("Number of Classes", len(CLASS_NAMES))
    with col3:
        st.metric("Model Uptime", get_model_uptime())
    
    st.markdown("---")
    
    st.subheader("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Features")
        st.markdown("""
        - **Real-time Prediction**: Upload audio and get instant predictions
        - **Batch Processing**: Process multiple audio files at once
        - **Data Visualization**: Explore dataset statistics and model performance
        - **Model Retraining**: Upload new data and retrain the model
        - **Performance Monitoring**: Track model metrics and uptime
        """)
    
    with col2:
        st.markdown("### Supported Sound Classes")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.markdown(f"{i}. **{class_name.replace('_', ' ').title()}**")
    
    st.markdown("---")
    
    st.subheader("Quick Start")
    st.markdown("""
    1. **Navigate to Predict** page to classify individual or batch audio files
    2. **Check Visualizations** to explore data insights and model performance
    3. **Upload new data** and trigger retraining to improve the model
    4. **Monitor** model performance and system health
    """)
    
    # Model info
    try:
        model_info = predictor.get_model_info()
        
        with st.expander("Technical Details"):
            st.write("**Model Information:**")
            info_data = {
                "Property": ["Model Type", "Number of Classes", "Feature Size", "Status"],
                "Value": [
                    model_info.get('model_type', 'Unknown'),
                    model_info.get('num_classes', 'Unknown'),
                    model_info.get('feature_size', 'Unknown'),
                    model_info.get('status', 'Unknown')
                ]
            }
            safe_table_display(info_data)
    except Exception as e:
        st.warning(f"Could not load model info: {e}")


def show_prediction_page(predictor):
    st.header("Sound Classification Prediction")
    
    tab1, tab2 = st.tabs(["Single File Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Upload a single audio file")
        uploaded_file = st.file_uploader(
            "Choose a WAV file", 
            type=['wav'],
            key="single_file"
        )
        
        if uploaded_file is not None:
            # Create temp directory
            temp_dir = './data/temp'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded file with unique name
            temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{uploaded_file.name}")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Predict", key="predict_single"):
                with st.spinner("Analyzing audio..."):
                    try:
                        result = predictor.predict(temp_path)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        result = None
                    finally:
                        # Always clean up temp file
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except Exception as cleanup_error:
                            st.warning(f"Could not clean up temp file: {cleanup_error}")
                
                if result and not result.get('error'):
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown(f"### Prediction: **{result['class_name'].replace('_', ' ').title()}**")
                    st.markdown(f"**Confidence**: {result['confidence']:.2%}")
                    
                    confidence = result['confidence']
                    if confidence > 0.8:
                        explanation = "High confidence prediction"
                    elif confidence > 0.6:
                        explanation = "Moderate confidence prediction"
                    else:
                        explanation = "Low confidence prediction"
                    st.markdown(f"**Explanation**: {explanation}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show top 3 predictions
                    st.subheader("Top 3 Predictions")
                    top_3 = result.get('all_predictions', [])
                    
                    for i, pred in enumerate(top_3, 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i}. {pred['class_name'].replace('_', ' ').title()}**")
                        with col2:
                            st.markdown(f"{pred['confidence']:.2%}")
                        st.progress(pred['confidence'])
                    
                    # Show all probabilities
                    with st.expander("View all class probabilities"):
                        probs = result.get('probabilities', {})
                        probs_df = pd.DataFrame([
                            {"Class": k.replace('_', ' ').title(), "Probability": f"{v:.4f}"}
                            for k, v in probs.items()
                        ]).sort_values('Probability', ascending=False)
                        safe_dataframe_display(probs_df, width='stretch')
                else:
                    error_msg = result.get('error', 'Unknown error') if result else "Prediction failed"
                    st.error(f"Failed to process audio file: {error_msg}")
    
    with tab2:
        st.subheader("Upload multiple audio files")
        uploaded_files = st.file_uploader(
            "Choose WAV files", 
            type=['wav'],
            accept_multiple_files=True,
            key="batch_files"
        )
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("Predict All", key="predict_batch"):
                temp_dir = './data/temp'
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save all files with unique names
                file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, f"batch_{int(time.time())}_{uploaded_file.name}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_path)
                
                # Batch predict
                with st.spinner(f"Processing {len(file_paths)} files..."):
                    try:
                        results = predictor.predict_batch(file_paths)
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")
                        results = None
                    finally:
                        # Clean up ALL temp files
                        for temp_path in file_paths:
                            try:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                            except Exception as cleanup_error:
                                st.warning(f"Could not clean up {os.path.basename(temp_path)}: {cleanup_error}")
                
                if results:
                    # Filter out errors
                    valid_results = [r for r in results if not r.get('error')]
                    
                    if valid_results:
                        st.success(f"Successfully processed {len(valid_results)} files!")
                        
                        # Show results table
                        results_df = pd.DataFrame([
                            {
                                "Filename": os.path.basename(r['file_path']),
                                "Predicted Class": r['class_name'].replace('_', ' ').title(),
                                "Confidence": f"{r['confidence']:.2%}"
                            }
                            for r in valid_results
                        ])
                        
                        safe_dataframe_display(results_df, width='stretch')
                        
                        # Calculate summary statistics
                        confidences = [r['confidence'] for r in valid_results]
                        predicted_classes = [r['class_name'] for r in valid_results]
                        
                        # Count predictions per class
                        class_counts = {}
                        for cls in predicted_classes:
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")
                        with col2:
                            st.metric("High Confidence", sum(1 for c in confidences if c > 0.8))
                        with col3:
                            st.metric("Low Confidence", sum(1 for c in confidences if c < 0.5))
                        
                        # Class distribution chart
                        st.subheader("Prediction Distribution")
                        class_dist = pd.DataFrame([
                            {"Class": k.replace('_', ' ').title(), "Count": v}
                            for k, v in class_counts.items()
                        ])
                        if not class_dist.empty:
                            st.bar_chart(class_dist.set_index('Class'))
                    else:
                        st.error("No valid predictions were made. Check the audio files.")
                else:
                    st.error("Batch prediction failed.")


def show_visualization_page(predictor):
    st.header(" Data Visualizations & Model Insights")
    
    # Check if we have saved data
    if not os.path.exists('./models/sound_classifier.pkl'):
        st.warning("No model data available. Please run the notebook first.")
        return
    
    try:
        # Load model for feature importance
        model = joblib.load('./models/sound_classifier.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    tab1, tab2, tab3 = st.tabs(["Dataset Analysis", "Model Performance", "Feature Importance"])
    
    with tab1:
        st.subheader("Dataset Statistics")
        
        # Check training data
        data_exists, existing_files, missing_files = check_training_data()
        if data_exists:
            try:
                X_train = np.load('./data/processed/X_train.npy')
                y_train = np.load('./data/processed/y_train.npy')
                
                st.write(f"Training samples: {X_train.shape[0]}")
                st.write(f"Feature size: {X_train.shape[1]}")
                st.write(f"Number of classes: {len(np.unique(y_train))}")
                
                # Class distribution
                st.subheader("Class Distribution")
                unique, counts = np.unique(y_train, return_counts=True)
                class_dist = pd.DataFrame({
                    'Class': [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}' for i in unique],
                    'Count': counts
                })
                st.bar_chart(class_dist.set_index('Class'))
                
            except Exception as e:
                st.error(f"Error loading training data: {e}")
        else:
            st.warning("Training data not available.")
            st.info("To create training data:")
            st.code("cd notebook\npython create_training_data.py")
            
            # Show class names
            st.subheader("Available Classes")
            for i, cls in enumerate(CLASS_NAMES, 1):
                st.write(f"{i}. {cls.replace('_', ' ').title()}")
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        # Show model info
        try:
            model_info = predictor.get_model_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Type", model_info.get('model_type', 'Unknown'))
                st.metric("Number of Classes", model_info.get('num_classes', 'Unknown'))
            with col2:
                st.metric("Feature Vector Size", model_info.get('feature_size', 'Unknown'))
        except Exception as e:
            st.error(f"Error getting model info: {e}")
        
        st.info("To see detailed performance metrics, please check the confusion matrix and classification report in the notebook.")
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Top N features
            top_n = st.slider("Number of top features to display", 10, 50, 20)
            
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_importances = importances[top_indices]
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(top_n), top_importances, color='orange', alpha=0.7)
            ax.set_title(f'Top {top_n} Most Important MFCC Features', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance Score')
            ax.set_xticks(range(top_n))
            ax.set_xticklabels([f'F{idx}' for idx in top_indices], rotation=45)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Importance", f"{np.max(importances):.6f}")
            with col2:
                st.metric("Mean Importance", f"{np.mean(importances):.6f}")
            with col3:
                st.metric("Std Dev", f"{np.std(importances):.6f}")
        else:
            st.warning("Feature importance not available for this model type.")


def show_upload_retrain_page(predictor):
    st.header(" Upload Data & Retrain Model")
    
    st.markdown("""
    Upload new audio files to improve the model. Files should be organized by class.
    """)
    
    tab1, tab2 = st.tabs(["Upload Data", "Trigger Retraining"])
    
    with tab1:
        st.subheader("Upload Training Data")
        
        # Select class
        selected_class = st.selectbox(
            "Select sound class for uploaded files",
            CLASS_NAMES,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            f"Upload WAV files for '{selected_class.replace('_', ' ').title()}'",
            type=['wav'],
            accept_multiple_files=True,
            key="upload_training"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files")
            
            if st.button(" Save Files", key="save_files"):
                # Create class directory
                class_dir = os.path.join(UPLOAD_DIR, selected_class)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save files
                saved_count = 0
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(class_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                
                st.success(f" Successfully saved {saved_count} files to {class_dir}!!!!")
        
        # Show uploaded data summary
        st.subheader("Uploaded Data Summary")
        
        upload_summary = {}
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(UPLOAD_DIR, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                if files:
                    upload_summary[class_name.replace('_', ' ').title()] = len(files)
        
        if upload_summary:
            summary_df = pd.DataFrame([
                {"Class": k, "Files": v}
                for k, v in upload_summary.items()
            ])
            safe_dataframe_display(summary_df, width='stretch')
            st.info(f"Total uploaded files: {sum(upload_summary.values())}")
        else:
            st.info("No files uploaded yet.")
    
    with tab2:
        st.subheader("Retrain Model")
        
        st.markdown("""
        **Complete Retraining Pipeline:**
        1.  Load original training data
        2.  Process new uploaded audio files  
        3.  Combine datasets
        4.  Retrain the model
        5.  Evaluate performance
        6.  Save the new model
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_exists = os.path.exists(MODEL_PATH)
            st.write(f"Model: {' Found' if model_exists else '❌ Missing'}")
        
        with col2:
            data_exists, existing_files, missing_files = check_training_data()
            st.write(f"Training Data: {' Found' if data_exists else '❌ Missing'}")
        
        with col3:
            upload_count = 0
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(UPLOAD_DIR, class_name)
                if os.path.exists(class_dir):
                    upload_count += len([f for f in os.listdir(class_dir) if f.endswith('.wav')])
            st.write(f"New Files: {' ' + str(upload_count) if upload_count > 0 else '❌ None'}")
        
        if not data_exists:
            st.error(" Training data is missing. Please create training data first.")
            
            if st.button(" Create Sample Training Data", key="create_sample_data"):
                with st.spinner("Creating sample training data..."):
                    success, message = create_missing_training_data()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            st.info("Missing files:")
            for missing_file in missing_files:
                st.write(f"- {missing_file}")
                
        elif not all([model_exists, data_exists, upload_count > 0]):
            st.error("Cannot start retraining. Please ensure:")
            if not model_exists:
                st.write("- Base model exists in ./models/")
            if not data_exists:
                st.write("- Original training data exists")
            if upload_count == 0:
                st.write("- New audio files are uploaded in the Upload Data tab")
        else:
            st.success(" All prerequisites met! Ready for retraining.")
            
            st.warning(" Retraining may take several minutes depending on data size.")
            
            if st.button(" Start Real Retraining", key="trigger_real_retrain"):
                with st.spinner("Starting complete retraining pipeline..."):
                    try:
                        # Initialize retrainer
                        retrainer = ModelRetrainer(MODEL_PATH)
                        
                        # Create progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Load original data
                        status_text.text(" Step 1/6: Loading original training data...")
                        progress_bar.progress(16)
                        time.sleep(1)
                        
                        # Step 2: Process new audio files
                        status_text.text(" Step 2/6: Processing new audio files...")
                        progress_bar.progress(32)
                        time.sleep(1)
                        
                        # Step 3: Combine datasets
                        status_text.text(" Step 3/6: Combining datasets...")
                        progress_bar.progress(48)
                        time.sleep(1)
                        
                        # Step 4: Retrain model (this is the actual retraining)
                        status_text.text(" Step 4/6: Retraining model...")
                        progress_bar.progress(64)
                        
                        # Perform actual retraining
                        results = retrainer.retrain(save_path=MODEL_PATH)
                        
                        # Step 5: Evaluate performance
                        status_text.text(" Step 5/6: Evaluating performance...")
                        progress_bar.progress(80)
                        time.sleep(1)
                        
                        # Step 6: Save model
                        status_text.text(" Step 6/6: Saving model...")
                        progress_bar.progress(100)
                        time.sleep(1)
                        
                        st.success(" Retraining completed successfully!")
                        
                        # Display real results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training Accuracy", f"{results['train_accuracy']:.4f}")
                        with col2:
                            st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
                        with col3:
                            st.metric("Training Time", f"{results['training_time']:.2f}s")
                        with col4:
                            st.metric("Total Samples", results['total_samples'])
                        
                        # Show detailed breakdown
                        with st.expander(" Detailed Results"):
                            st.write(f"**Original Dataset:** {results['original_samples']} samples")
                            st.write(f"**New Uploaded Files:** {results['new_samples']} samples") 
                            st.write(f"**Combined Dataset:** {results['total_samples']} samples")
                            st.write(f"**Improvement:** +{results['new_samples']} samples ({results['new_samples']/results['original_samples']*100:.1f}% increase)")
                            
                        # Clear uploaded files after successful retraining
                        for class_name in CLASS_NAMES:
                            class_dir = os.path.join(UPLOAD_DIR, class_name)
                            if os.path.exists(class_dir):
                                for file in os.listdir(class_dir):
                                    if file.endswith('.wav'):
                                        os.remove(os.path.join(class_dir, file))
                        st.info("Uploaded files have been processed and cleared.")
                        
                    except Exception as e:
                        st.error(f" Error during retraining: {str(e)}")
                        st.info("Please ensure:")
                        st.write("- Original training data exists in ./data/processed/")
                        st.write("- Uploaded audio files are valid WAV files")
                        st.write("- Model file is compatible")


def show_monitoring_page(predictor):
    """Monitoring page"""
    st.header("📈 Model Monitoring & Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Status", " Online")
    with col2:
        st.metric("Model Uptime", get_model_uptime())
    with col3:
        st.metric("API Status", " Ready")
    with col4:
        st.metric("Last Updated", datetime.now().strftime('%H:%M:%S'))
    
    st.markdown("---")
    
    # System health
    st.subheader("System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Files")
        model_exists = os.path.exists(MODEL_PATH)
        preprocessor_exists = os.path.exists(PREPROCESSOR_PATH)
        
        st.write(f" Model file: {'Found' if model_exists else '❌ Missing'}")
        st.write(f" Preprocessor file: {'Found' if preprocessor_exists else '❌ Missing'}")
        
        if model_exists:
            model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.write(f"Model size: {model_size:.2f} MB")
    
    with col2:
        st.markdown("### Directory Structure")
        dirs = ['./models', './data', './src', './api', './ui']
        for dir_path in dirs:
            exists = os.path.exists(dir_path)
            st.write(f"{'' if exists else '❌'} {dir_path}")
    
    st.markdown("---")
    
    # Model info
    st.subheader("Model Information")
    try:
        model_info = predictor.get_model_info()
        
        info_df = pd.DataFrame([
            {"Property": "Model Type", "Value": model_info.get('model_type', 'Unknown')},
            {"Property": "Number of Classes", "Value": model_info.get('num_classes', 'Unknown')},
            {"Property": "Feature Vector Size", "Value": model_info.get('feature_size', 'Unknown')},
            {"Property": "Status", "Value": model_info.get('status', 'Unknown')},
        ])
        
        safe_table_display(info_df)
    except Exception as e:
        st.error(f"Error getting model info: {e}")
    
    # Refresh button
    if st.button(" Refresh Status"):
        st.rerun()


if __name__ == "__main__":
    main()