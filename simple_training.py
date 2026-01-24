"""
Simplified training script for quick start
SpeechBrain pre-trained model version
"""
import os
import torch
from speechbrain.inference.ASR import EncoderDecoderASR
import pandas as pd
from jiwer import wer, cer
from pathlib import Path
from tqdm import tqdm


class SimpleTIMITTrainer:
    """Simplified trainer using a pre-trained model"""
    
    def __init__(self, data_dir="data/manifests"):
        self.data_dir = Path(data_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
    def load_pretrained_model(self):
        print("\nLoading pre-trained model...")
        
        try:
            # Pre-trained model on LibriSpeech
            self.asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-crdnn-rnnlm-librispeech",
                savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
                run_opts={"device": self.device}
            )
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Downloading model...")
            return False
    
    def fine_tune_on_timit(self, train_csv, epochs=10):
        print(f"\nFine-tuning on TIMIT (SA1/SA2) - {epochs} epochs")
        
        # Load data
        train_df = pd.read_csv(train_csv)
        print(f"Training files: {len(train_df)}")
        
        print("\nNote: For actual fine-tuning, use the complete script")
        print("This script performs direct evaluation of the pre-trained model")
        
        return True
    
    def evaluate(self, test_csv):
        print(f"\nEvaluating on test set...")
        
        # Load test data
        test_df = pd.read_csv(test_csv)
        print(f"Test files: {len(test_df)}")
        
        references = []
        hypotheses = []
        
        # Evaluate each file
        print("\nTranscribing...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            audio_path = row['audio_path']
            reference = row['transcription'].lower()
            
            try:
                # Transcribe
                hypothesis = self.asr_model.transcribe_file(audio_path)
                hypothesis = hypothesis.lower()
                
                references.append(reference)
                hypotheses.append(hypothesis)
                
            except Exception as e:
                print(f"\nError for {audio_path}: {e}")
        
        # Calculate metrics
        if len(references) > 0:
            wer_score = wer(references, hypotheses) * 100
            cer_score = cer(references, hypotheses) * 100
            
            print(f"\n" + "="*60)
            print(f"EVALUATION RESULTS")
            print("="*60)
            print(f"Files evaluated: {len(references)}/{len(test_df)}")
            print(f"WER (Word Error Rate): {wer_score:.2f}%")
            print(f"CER (Character Error Rate): {cer_score:.2f}%")
            print("="*60)
            
            print(f"\nTranscription examples:")
            for i in range(min(3, len(references))):
                print(f"\nFile #{i+1}:")
                print(f"  Reference:    {references[i]}")
                print(f"  Hypothesis:   {hypotheses[i]}")
            
            return {
                'wer': wer_score,
                'cer': cer_score,
                'n_files': len(references)
            }
        else:
            print("No files could be transcribed!")
            return None
    
    def transcribe_single(self, audio_path):
        print(f"\nTranscribing: {audio_path}")
        
        try:
            transcription = self.asr_model.transcribe_file(audio_path)
            print(f"Result: {transcription}")
            return transcription
        except Exception as e:
            print(f"Error: {e}")
            return None


def main():
    print("="*60)
    print("ASR TIMIT - Simplified Training")
    print("="*60)
    
    # Initialize trainer
    trainer = SimpleTIMITTrainer(data_dir="data/manifests")
    
    # Load pre- trained model
    if not trainer.load_pretrained_model():
        print("\nUnable to load pre-trained model")
        print("Check your internet connection and try again")
        return
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Transcribe an audio file")
        print("2. Evaluate on TIMIT test set")
        print("3. Exit")
        
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == "1":
            audio_path = input("Audio file path: ").strip()
            if Path(audio_path).exists():
                trainer.transcribe_single(audio_path)
            else:
                print(f"File not found: {audio_path}")
        
        elif choice == "2":
            test_csv = "../data/manifests/test_sa.csv"
            if Path(test_csv).exists():
                results = trainer.evaluate(test_csv)
                
                # Save results
                if results:
                    output_dir = "results/simple_evaluation"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    import json
                    with open(f"{output_dir}/results.json", "w") as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"\nResults saved to: {output_dir}/results.json")
            else:
                print(f"Test file not found: {test_csv}")
                print("Run first: python data_preparation.py")
        
        elif choice == "3":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Choose 1, 2, or 3.")


if __name__ == "__main__":
    main()