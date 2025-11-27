#  Sound Classification ML Project

A machine learning system that classifies environmental sounds into 10 categories using MFCC features and Random Forest classifier. The project includes a web interface, REST API, and comprehensive monitoring capabilities.

##  Project Overview

This project demonstrates a complete ML pipeline for sound classification:
- **10 Sound Classes**: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- **Full-stack Application**: Streamlit UI, Flask API, and production deployment
- **Model Management**: Training, retraining, and monitoring capabilities
- **Load Testing**: Performance testing with Locust

##  Quick Start

### Prerequisites

- Python 3.8+
- Git
- (Optional) Docker

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd sound-classification-ml

# Create virtual environment (Windows)
python -m venv sound_env
sound_env\Scripts\activate

# Or on Mac/Linux
python -m venv sound_env
source sound_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Training Data

```bash
# Generate the model and training data
cd notebook
python create_compatible_model.py
cd ..
```

### 3. Run the Application

#### Option A: Streamlit Web Interface
```bash
python -m streamlit run ui/streamlit_app.py
```
Visit: `http://localhost:8501`

#### Option B: Flask API
```bash
python api/app.py
```
API available at: `http://localhost:5000`

#### Option C: Docker (All Services)
```bash
docker-compose up -d --scale web=3
```
- Streamlit UI: `http://localhost:8501`
- Flask API: `http://localhost:5000`
- Load Testing: `http://localhost:8089`

##  Project Structure

```
sound-classification-ml/
├──  api/                 # Flask REST API
│   └── app.py             # Main API application
├──  data/               # Dataset and processed data
│   ├── processed/         # Processed features
│   └── upload/           # User uploads for retraining
├──  models/             # Trained models
│   └── sound_classifier.pkl
├──  notebook/           # Jupyter notebooks & model creation
│   └── create_compatible_model.py
├──  src/               # Core ML code
│   ├── model.py          # Model training & retraining
│   ├── preprocessing.py  # Audio feature extraction
│   └── prediction.py     # Prediction logic
├──  ui/                # Streamlit web interface
│   └── streamlit_app.py  # Main UI application
├──  load_test/         # Load testing with Locust
│   └── locustfile.py     # Load test scenarios
├── 📄 docker-compose.yml # Multi-container setup
├── 📄 requirements.txt   # Python dependencies
└── 📄 README.md          # This file
```

##  Using the Application

### Web Interface Features

1. **Home Dashboard**
   - System overview and model status
   - Performance metrics and uptime monitoring

2. **Sound Prediction**
   - Upload individual audio files (WAV format)
   - Batch processing for multiple files
   - Real-time confidence scores and visualizations

3. **Model Management**
   - Upload new training data
   - Trigger model retraining
   - Monitor training progress and results

4. **Visualizations**
   - Dataset statistics and class distribution
   - Feature importance analysis
   - Model performance metrics

### API Endpoints

```http
GET  /health     # Health check and system status
POST /predict    # Sound classification
GET  /metrics    # Performance metrics
POST /batch_predict  # Batch predictions
```

#### Example API Usage

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_encoded_audio", "model_type": "random_forest"}'

# Get metrics
curl http://localhost:5000/metrics
```

##  Deployment

### 1. Railway Deployment (Recommended)

Railway provides easy deployment with automatic scaling:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy from current directory
railway up
```

**Manual Deployment Steps:**
1. Go to [Railway](https://railway.app)
2. Connect your GitHub repository
3. Deploy automatically from main branch
4. Your app will be available at: `https://your-project-name.up.railway.app`

### 2. Heroku Deployment

```bash
# Install Heroku CLI
# Login and create app
heroku create your-sound-classification-app

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main
```

### 3. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d --build

# Scale API instances
docker-compose up -d --scale web=3

# View logs
docker-compose logs -f
```

##  Load Testing

### Using Locust

```bash
# Run load tests locally
cd load_test
locust -f locustfile.py --host=http://localhost:5000

# Access Locust web interface at: http://localhost:8089

# Or run headless tests
locust -f locustfile.py --host=http://localhost:5000 --headless -u 100 -r 10 -t 10m
```

### Performance Testing Scripts

```bash
# Test different load scenarios
./run_load_test_flexible.sh

# Monitor performance in real-time
./monitor_production.sh
```

##  Configuration

### Environment Variables

```bash
# API Configuration
FLASK_ENV=production
MODEL_PATH=./models/sound_classifier.pkl
PORT=5000

# Deployment
RAILWAY_URL=https://your-app.up.railway.app
```

### Model Configuration

The default model uses:
- **Algorithm**: Random Forest
- **Features**: MFCC (Mel-frequency cepstral coefficients)
- **Classes**: 10 environmental sounds
- **Training Data**: ESC-50 dataset subset

## 🛠️ Development

### Adding New Sound Classes

1. Update `CLASS_NAMES` in `src/preprocessing.py`
2. Add training data in appropriate format
3. Retrain the model

### Model Retraining

1. Upload new audio files through the web interface
2. Organize files by class in `data/upload/`
3. Use the "Upload & Retrain" page to trigger retraining
4. Monitor training progress and results

### Customizing Feature Extraction

Modify `src/preprocessing.py`:
```python
# Adjust MFCC parameters
n_mfcc=40,           # Number of MFCC coefficients
sr=22050,            # Sample rate
n_fft=2048,          # FFT window size
hop_length=512,      # Sliding window hop
```

##  Monitoring & Logs

### Application Logs

```bash
# Streamlit logs
python -m streamlit run ui/streamlit_app.py

# API logs
python api/app.py

# Docker logs
docker-compose logs -f web
```

### Performance Metrics

- Response times and throughput
- Model accuracy and confidence scores
- System resource usage
- Error rates and failure analysis

##  Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Recreate the model
   cd notebook && python create_compatible_model.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Audio Processing Issues**
   - Ensure audio files are in WAV format
   - Check file permissions and paths
   - Verify librosa and soundfile installations

4. **Port Conflicts**
   ```bash
   # Change default ports
   python api/app.py --port 5001
   streamlit run ui/streamlit_app.py --server.port 8502
   ```

### Getting Help

1. Check the logs for detailed error messages
2. Verify all prerequisite steps are completed
3. Ensure training data exists in `data/processed/`
4. Confirm model file exists in `models/sound_classifier.pkl`

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🎓 Academic Reference

This project uses techniques from:
- Audio feature extraction with Librosa
- Machine learning with Scikit-learn
- Model deployment and monitoring best practices

---

**Ready to classify some sounds?** 🎵 Start with the [Quick Start](#-quick-start) section above!