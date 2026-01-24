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
        """
        Args:
            sample_rate: Fréquence d'échantillonnage (16kHz pour ASR)
        """
        self.sample_rate = sample_rate
        # Initialiser OpenSMILE pour PLP
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
    def load_audio(self, audio_path):
        """
        Charge un fichier audio et le rééchantillonne si nécessaire
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def extract_mfcc(self, audio_path, n_mfcc=13, n_fft=512, hop_length=160):
        """
        Extraction des MFCC avec Librosa
        
        Args:
            audio_path: Chemin vers le fichier audio
            n_mfcc: Nombre de coefficients MFCC (13 par défaut)
            n_fft: Taille de la FFT
            hop_length: Pas de la fenêtre glissante
            
        Returns:
            Matrice MFCC (n_mfcc, n_frames)
        """
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
        """
        Extraction des PLP avec OpenSMILE
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Features PLP
        """
        try:
            features = self.smile.process_file(audio_path)
            return features.values
        except Exception as e:
            print(f"Erreur extraction PLP pour {audio_path}: {e}")
            return None
    
    def extract_all_features(self, audio_path):
        """
        Extrait MFCC et PLP pour un fichier audio
        
        Returns:
            Dictionnaire avec 'mfcc' et 'plp'
        """
        features = {}
        
        # MFCC
        features['mfcc'] = self.extract_mfcc(audio_path)
        
        # PLP
        features['plp'] = self.extract_plp(audio_path)
        
        return features
    
    def extract_mel_spectrogram(self, audio_path, n_mels=80):
        """
        Extraction du spectrogramme Mel (pour approche end-to-end)
        
        Args:
            audio_path: Chemin vers le fichier audio
            n_mels: Nombre de bandes Mel
            
        Returns:
            Spectrogramme Mel (n_mels, n_frames)
        """
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
        """
        Sauvegarde les features extraites
        """
        np.save(output_path, features)
        print(f"Features sauvegardées: {output_path}")


# Batch processing pour un dataset complet
def batch_extract_features(manifest_csv, output_dir, feature_type='mfcc'):
    """
    Extrait les features pour tous les fichiers d'un manifest
    
    Args:
        manifest_csv: Fichier CSV avec les chemins audio
        output_dir: Dossier de sortie pour les features
        feature_type: 'mfcc', 'plp', ou 'mel'
    """
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
    # Exemple d'utilisation
    extractor = FeatureExtractor()
    
    # Test sur un fichier
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