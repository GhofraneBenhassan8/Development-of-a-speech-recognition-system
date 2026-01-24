"""
Extraction des caractéristiques acoustiques: MFCC et PLP
"""
import librosa
import numpy as np
import opensmile
import soundfile as sf
from pathlib import Path

class FeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Initialiser OpenSMILE pour PLP
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def extract_mfcc(self, audio_path, n_mfcc=13, n_fft=512, hop_length=160):
        audio, sr = self.load_audio(audio_path)
        
        # Extraction MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Delta et Delta-Delta (dérivées première et seconde)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Concaténer MFCC + Delta + Delta2
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        
        return features.T  # Transposer pour avoir (n_frames, n_features)
    
    def extract_plp(self, audio_path):
        try:
            features = self.smile.process_file(audio_path)
            return features.values
        except Exception as e:
            print(f"Erreur extraction PLP pour {audio_path}: {e}")
            return None
    
    def extract_all_features(self, audio_path):
        features = {}
        
        # MFCC
        features['mfcc'] = self.extract_mfcc(audio_path)
        
        # PLP
        features['plp'] = self.extract_plp(audio_path)
        
        return features
    
    def extract_mel_spectrogram(self, audio_path, n_mels=80):
        audio, sr = self.load_audio(audio_path)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=512,
            hop_length=160
        )
        
        # Conversion en échelle log
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db.T  # (n_frames, n_mels)
    
    def save_features(self, features, output_path):
        np.save(output_path, features)
        print(f"Features sauvegardées: {output_path}")


# Batch processing pour un dataset complet
def batch_extract_features(manifest_csv, output_dir, feature_type='mfcc'):
    import pandas as pd
    
    df = pd.DataFrame(manifest_csv)
    extractor = FeatureExtractor()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        filename = Path(audio_path).stem
        
        try:
            if feature_type == 'mfcc':
                features = extractor.extract_mfcc(audio_path)
            elif feature_type == 'plp':
                features = extractor.extract_plp(audio_path)
            elif feature_type == 'mel':
                features = extractor.extract_mel_spectrogram(audio_path)
            else:
                raise ValueError(f"Feature type non reconnu: {feature_type}")
            
            output_path = Path(output_dir) / f"{filename}_{feature_type}.npy"
            extractor.save_features(features, output_path)
            
            if (idx + 1) % 50 == 0:
                print(f"Traité {idx + 1}/{len(df)} fichiers")
                
        except Exception as e:
            print(f"Erreur pour {audio_path}: {e}")
    
    print(f"\nExtraction terminée! Features dans {output_dir}")


# Test
if __name__ == "__main__":
    extractor = FeatureExtractor()
    test_audio = "../data/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV"
    
    if Path(test_audio).exists():
        print("Extraction MFCC...")
        mfcc = extractor.extract_mfcc(test_audio)
        print(f"MFCC shape: {mfcc.shape}")
        
        print("\nExtraction Mel Spectrogram...")
        mel = extractor.extract_mel_spectrogram(test_audio)
        print(f"Mel shape: {mel.shape}")
    else:
        print(f"Fichier de test non trouvé: {test_audio}")
        print("Remplacez par un chemin valide vers un fichier TIMIT")