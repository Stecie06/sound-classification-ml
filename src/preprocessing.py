import numpy as np
import librosa
import os

CLASS_NAMES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", 
               "drilling", "engine_idling", "gun_shot", "jackhammer", 
               "siren", "street_music"]

class AudioPreprocessor:
    
    def __init__(self, sr=22050, n_mfcc=40, max_len=174):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        print(f"AudioPreprocessor initialized:")
        print(f"   Sample Rate: {self.sr}Hz")
        print(f"   MFCC Coefficients: {self.n_mfcc}")
        print(f"   Max Length: {self.max_len} frames")

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, duration=5)

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)

            if mfccs.shape[1] < self.max_len:
                pad_width = self.max_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_len]

            return mfccs.flatten()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def batch_process(self, audio_files, labels):
        features = []
        valid_labels = []
        valid_files = []

        print(f"Processing {len(audio_files)} audio files...")
        for i, (file_path, label) in enumerate(zip(audio_files, labels)):
            if i % 20 == 0 and i > 0:
                print(f"   Processed {i}/{len(audio_files)} files...")

            feature = self.extract_features(file_path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(label)
                valid_files.append(file_path)

        print(f"Completed! Processed {len(features)}/{len(audio_files)} files successfully")
        return np.array(features), np.array(valid_labels), valid_files

    def get_feature_size(self):
        return self.n_mfcc * self.max_len


def load_dataset(data_dir, preprocessor):
    audio_files = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        for audio_file in os.listdir(class_dir):
            if audio_file.endswith('.wav'):
                audio_files.append(os.path.join(class_dir, audio_file))
                labels.append(class_idx)
    
    print(f"Found {len(audio_files)} audio files across {len(CLASS_NAMES)} classes")
    
    # Process all files
    features, labels, valid_files = preprocessor.batch_process(audio_files, labels)
    
    return features, labels, valid_files


if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    print(f"Feature vector size: {preprocessor.get_feature_size()}")