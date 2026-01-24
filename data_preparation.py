"""
Préparation des données TIMIT - Focus sur SA1 et SA2
"""
import os
import json
import pandas as pd
from pathlib import Path

class TIMITDataPreparation:
    def __init__(self, timit_root):
        self.timit_root = Path(timit_root)
        self.train_path = self.timit_root / "TRAIN"
        self.test_path = self.timit_root / "TEST"
        
    def get_sa_files(self, split='train'):
        data = []
        base_path = self.train_path if split == 'train' else self.test_path
        
        # Parcourir tous les dialectes (DR1, DR2, ...DR8)
        for dialect_dir in base_path.iterdir():
            if not dialect_dir.is_dir() or not dialect_dir.name.startswith('DR'):
                continue
                
            # Parcourir tous les locuteurs
            for speaker_dir in dialect_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                    
                speaker_id = speaker_dir.name
                
                # Chercher les fichiers SA1 et SA2
                for sentence_file in speaker_dir.glob('SA*.WAV'):
                    sentence_id = sentence_file.stem
                    txt_file = sentence_file.with_suffix('.TXT')
                    
                    if txt_file.exists():
                        # Lire la transcription
                        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            # Format TIMIT: start_sample end_sample transcription
                            transcription = ' '.join(lines[0].split()[2:])
                        
                        data.append({
                            'audio_path': str(sentence_file),
                            'transcription': transcription,
                            'speaker_id': speaker_id,
                            'sentence_id': sentence_id,
                            'dialect': dialect_dir.name
                        })
        
        return data
    
    def create_manifest(self, output_dir='../data/manifests'):
        # Create the output directory
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Collecter les données SA1/SA2
        print("Collecte des fichiers SA1 et SA2 (train)...")
        train_data = self.get_sa_files(split='train')
        
        print("Collecte des fichiers SA1 et SA2 (test)...")
        test_data = self.get_sa_files(split='test')
        
        # Sauvegarder en CSV
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        train_csv = output_path / 'train_sa.csv'
        test_csv = output_path / 'test_sa.csv'
        
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        print(f"Train SA1/SA2: {len(train_data)} fichiers")
        print(f"Test SA1/SA2: {len(test_data)} fichiers")
        print(f"Manifests sauvegardés dans {output_path}")
        
        # Sauvegarder aussi en JSON pour SpeechBrain
        with open(output_path / 'train_sa.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
            
        with open(output_path / 'test_sa.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
            
        return train_df, test_df
    
    def get_statistics(self, df):
        print("\n=== Statistiques===")
        print(f"Nombre total de fichiers: {len(df)}")
        print(f"Nombre de locuteurs uniques: {df['speaker_id'].nunique()}")
        print(f"Nombre de dialectes: {df['dialect'].nunique()}")
        print(f"\nRépartition par phrase:")
        print(df['sentence_id'].value_counts())
        print(f"\nRépartition par dialecte:")
        print(df['dialect'].value_counts())

if __name__ == "__main__":
    TIMIT_PATH = r"D:\timit_asr_project\data\TIMIT"
    
    prep = TIMITDataPreparation(TIMIT_PATH)
    train_df, test_df = prep.create_manifest()
    
    print("\n=== TRAIN SET ===")
    prep.get_statistics(train_df)
    
    print("\n=== TEST SET ===")
    prep.get_statistics(test_df)