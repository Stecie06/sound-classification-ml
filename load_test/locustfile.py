from locust import HttpUser, task, between
import os
import random
import time

class SoundClassificationUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        print("🚀 User started testing")
        
    def get_test_audio(self):
        """Get test audio file content"""
        test_files = [
            './data/test/test_audio.wav',
            './data/test/audio.wav', 
            './data/upload/sample.wav'  # fallback paths
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        # If no file found, create minimal WAV header
        print("⚠️  No test audio files found, using dummy data")
        return b'RIFF____WAVEfmt ____________________data____'
    
    @task(3)
    def health_check(self):
        with self.client.get("/health", catch_response=True, name="Health Check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2) 
    def home_page(self):
        self.client.get("/", name="Home Page")
    
    @task(2)
    def model_info(self):
        self.client.get("/model_info", name="Model Info")
    
    @task(5)
    def predict_audio(self):
        audio_data = self.get_test_audio()
        
        files = {'file': ('test_audio.wav', audio_data, 'audio/wav')}
        
        with self.client.post("/predict", 
                            files=files, 
                            catch_response=True,
                            name="Single Prediction") as response:
            
            if response.status_code == 200:
                try:
                    # Try to parse JSON to verify valid response
                    result = response.json()
                    if 'predicted_class' in result:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

class QuickTestUser(HttpUser):
    """Lightweight user for quick testing"""
    wait_time = between(0.5, 2)
    
    @task(10)
    def health_check(self):
        self.client.get("/health", name="Quick Health Check")
    
    @task(1)
    def other_endpoints(self):
        self.client.get("/", name="Quick Home")
        self.client.get("/model_info", name="Quick Model Info")