

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


TIMIT_ROOT =  r"D:\TIMIT\TIMIT"  # Change this!

# Reference sentences
SA_SENTENCES = {
    'SA1': "She had your dark suit in greasy wash water all year",
    'SA2': "Don't ask me to carry an oily rag like that"
}


def scan_timit_sa_files(timit_root):
    """Scan TIMIT directory and collect all SA1/SA2 files"""
    
    sa_files = []
    
    for split in ['TRAIN', 'TEST']:
        split_path = Path(timit_root) / split
        
        if not split_path.exists():
            print(f" Warning: {split_path} does not exist!")
            continue
        
        # Walk through dialect regions (DR1-DR8)
        for dr_folder in split_path.glob('DR*'):
            # Walk through speaker folders
            for speaker_folder in dr_folder.glob('*'):
                if speaker_folder.is_dir():
                    # Look for SA1 and SA2 files
                    for sa_file in ['SA1.WAV', 'SA2.WAV']:
                        wav_path = speaker_folder / sa_file
                        
                        if wav_path.exists():
                            # Extract metadata from path
                            speaker_id = speaker_folder.name
                            dialect = dr_folder.name
                            sentence_id = sa_file.replace('.WAV', '')
                            
                            sa_files.append({
                                'split': split,
                                'dialect': dialect,
                                'speaker_id': speaker_id,
                                'sentence_id': sentence_id,
                                'file_path': str(wav_path),
                                'sentence_text': SA_SENTENCES[sentence_id]
                            })
    
    return pd.DataFrame(sa_files)


def print_dataset_summary(df):
    """Print comprehensive dataset statistics"""
    
    print("=" * 80)
    print("TIMIT SA1/SA2 DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\n Total audio files found: {len(df)}")
    print(f"   - Train split: {len(df[df['split'] == 'TRAIN'])}")
    print(f"   - Test split: {len(df[df['split'] == 'TEST'])}")
    
    print(f"\n Unique speakers: {df['speaker_id'].nunique()}")
    print(f"   - Expected: 630 (462 train + 168 test)")
    
    print(f"\n Sentence distribution:")
    print(df['sentence_id'].value_counts())
    
    print(f"\n Dialect regions:")
    print(df['dialect'].value_counts().sort_index())
    
    print("\n Reference sentences:")
    for sid, text in SA_SENTENCES.items():
        print(f"   {sid}: '{text}'")
    
    print("\n" + "=" * 80)


def analyze_audio_file(file_path):
    """Extract basic audio properties"""
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Calculate properties
        duration = librosa.get_duration(y=y, sr=sr)
        rms_energy = np.sqrt(np.mean(y**2))
        zero_crossing_rate = np.mean(librosa.zero_crossings(y))
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'n_samples': len(y),
            'rms_energy': rms_energy,
            'zero_crossing_rate': zero_crossing_rate,
            'max_amplitude': np.max(np.abs(y)),
            'min_amplitude': np.min(y),
            'mean_amplitude': np.mean(y)
        }
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None


def analyze_dataset_audio(df, n_samples=50):
    """Analyze audio properties for a sample of files"""
    
    print(f"\n Analyzing audio properties (sample of {n_samples} files)...")
    
    # Sample files
    sample_df = df.sample(min(n_samples, len(df)), random_state=42)
    
    audio_stats = []
    for idx, row in sample_df.iterrows():
        stats = analyze_audio_file(row['file_path'])
        if stats:
            stats.update({
                'sentence_id': row['sentence_id'],
                'split': row['split'],
                'dialect': row['dialect']
            })
            audio_stats.append(stats)
    
    return pd.DataFrame(audio_stats)


def plot_audio_distributions(audio_df):
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('TIMIT SA Files - Audio Properties Distribution', fontsize=16, fontweight='bold')
    
    # Duration distribution
    axes[0, 0].hist(audio_df['duration'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Duration (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Audio Duration Distribution')
    axes[0, 0].axvline(audio_df['duration'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {audio_df["duration"].mean():.2f}s')
    axes[0, 0].legend()
    
    # RMS Energy distribution
    axes[0, 1].hist(audio_df['rms_energy'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('RMS Energy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('RMS Energy Distribution')
    
    # Max Amplitude distribution
    axes[0, 2].hist(audio_df['max_amplitude'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 2].set_xlabel('Max Amplitude')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Maximum Amplitude Distribution')
    
    # Duration by sentence
    audio_df.boxplot(column='duration', by='sentence_id', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Sentence')
    axes[1, 0].set_ylabel('Duration (seconds)')
    axes[1, 0].set_title('Duration by Sentence (SA1 vs SA2)')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)
    
    # Duration by split
    audio_df.boxplot(column='duration', by='split', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Split')
    axes[1, 1].set_ylabel('Duration (seconds)')
    axes[1, 1].set_title('Duration by Split (Train vs Test)')
    
    # RMS Energy by dialect
    sns.boxplot(data=audio_df, x='dialect', y='rms_energy', ax=axes[1, 2])
    axes[1, 2].set_xlabel('Dialect Region')
    axes[1, 2].set_ylabel('RMS Energy')
    axes[1, 2].set_title('Energy by Dialect Region')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def visualize_sample_audio(file_path, sentence_text):    
    y, sr = librosa.load(file_path, sr=None)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Audio Visualization\n"{sentence_text}"', fontsize=14, fontweight='bold')
    
    # 1. Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # 3. Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axes[2], format='%+2.0f dB')
    
    # 4. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[3])
    axes[3].set_title('MFCCs')
    axes[3].set_ylabel('MFCC Coefficient')
    fig.colorbar(img, ax=axes[3])
    
    plt.tight_layout()
    return fig


def compare_sa1_sa2(df):
    
    # Get one example of each
    sa1_file = df[df['sentence_id'] == 'SA1'].iloc[0]['file_path']
    sa2_file = df[df['sentence_id'] == 'SA2'].iloc[0]['file_path']
    
    # Load both
    y1, sr1 = librosa.load(sa1_file, sr=None)
    y2, sr2 = librosa.load(sa2_file, sr=None)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Comparison: SA1 vs SA2', fontsize=16, fontweight='bold')
    
    # SA1 Waveform
    librosa.display.waveshow(y1, sr=sr1, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title(f'SA1 Waveform\n"{SA_SENTENCES["SA1"]}"')
    axes[0, 0].set_xlabel('Time (s)')
    
    # SA2 Waveform
    librosa.display.waveshow(y2, sr=sr2, ax=axes[0, 1], color='red')
    axes[0, 1].set_title(f'SA2 Waveform\n"{SA_SENTENCES["SA2"]}"')
    axes[0, 1].set_xlabel('Time (s)')
    
    # SA1 Mel Spectrogram
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
    S1_db = librosa.amplitude_to_db(S1, ref=np.max)
    librosa.display.specshow(S1_db, sr=sr1, x_axis='time', y_axis='mel', ax=axes[1, 0])
    axes[1, 0].set_title('SA1 Mel Spectrogram')
    
    # SA2 Mel Spectrogram
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)
    S2_db = librosa.amplitude_to_db(S2, ref=np.max)
    librosa.display.specshow(S2_db, sr=sr2, x_axis='time', y_axis='mel', ax=axes[1, 1])
    axes[1, 1].set_title('SA2 Mel Spectrogram')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    
    print("\n Starting TIMIT SA Dataset Exploration...\n")
    
    # Step 1: Scan dataset
    print(" Scanning TIMIT directory structure...")
    df = scan_timit_sa_files(TIMIT_ROOT)
    
    if df.empty:
        print(" No SA files found! Please check your TIMIT_ROOT path.")
        print(f"   Current path: {TIMIT_ROOT}")
        print("\n Tips:")
        print("   - Make sure TIMIT is extracted")
        print("   - Path should point to folder containing TRAIN/ and TEST/")
        print("   - Check folder names are uppercase (TRAIN not train)")
        exit(1)
    
    # Step 2: Print summary
    print_dataset_summary(df)
    
    # Step 3: Analyze audio
    audio_df = analyze_dataset_audio(df, n_samples=100)
    
    print("\n Audio Statistics:")
    print(audio_df[['duration', 'rms_energy', 'max_amplitude']].describe())
    
    # Step 4: Generate visualizations
    print("\n Generating visualizations...")
    
    # Distribution plots
    fig1 = plot_audio_distributions(audio_df)
    plt.savefig('timit_audio_distributions.png', dpi=300, bbox_inches='tight')
    print(" Saved: timit_audio_distributions.png")
    
    # Detailed single file
    sample_file = df.iloc[0]
    fig2 = visualize_sample_audio(sample_file['file_path'], sample_file['sentence_text'])
    plt.savefig('timit_sample_audio_detailed.png', dpi=300, bbox_inches='tight')
    print("  Saved: timit_sample_audio_detailed.png")
    
    # SA1 vs SA2 comparison
    fig3 = compare_sa1_sa2(df)
    plt.savefig('timit_sa1_vs_sa2_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: timit_sa1_vs_sa2_comparison.png")
    
    # Save dataset info to CSV
    df.to_csv('timit_sa_files_inventory.csv', index=False)
    print(" Saved: timit_sa_files_inventory.csv")
    
    audio_df.to_csv('timit_audio_statistics.csv', index=False)
    print(" Saved: timit_audio_statistics.csv")
    
    print("\n" + "="*80)
    print(" Exploration complete!")
    print("="*80)
    print("\n Next steps:")
    print("   1. Review the generated visualizations")
    print("   2. Check the CSV files for detailed statistics")
    print("   3. Listen to some audio samples")
    print("   4. Proceed to feature extraction with OpenSMILE and Librosa")
    
    plt.show()
