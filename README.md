# Speech Recognition System with SpeechBrain

An Automatic Speech Recognition (ASR) system built using SpeechBrain framework, trained on the TIMIT dataset with a focus on SA1/SA2 calibration sentences.

## Project Overview

This project develops an end-to-end speech recognition system achieving **95% accuracy** (5.00% WER) on TIMIT test data. The system features a CNN-BiLSTM architecture with CTC loss and includes a real-time transcription GUI.

### Key Results
- **Word Error Rate (WER)**: 5.00%
- **Character Error Rate (CER)**: 5.05%
- **Training Time**: 6 epochs (with early stopping)
- **Improvement over baseline**: 78% reduction in WER

## Features

- **Deep Learning Architecture**: CNN encoder + 4-layer bidirectional LSTM + CTC loss
- **Advanced Tokenization**: Byte Pair Encoding (BPE) with 120 tokens
- **Real-time Transcription**: GUI application supporting:
  - Live microphone recording
  - Audio file upload (WAV, MP3, FLAC)
  - Instant transcription with timestamps
- **Comprehensive Pipeline**: Data preparation, feature extraction, training, and evaluation

## Architecture

```
Audio Input (WAV)
    ↓
Mel Spectrogram (80 bands)
    ↓
CNN Encoder (3 layers, 256 channels)
    ↓
Batch Normalization
    ↓
BiLSTM (4 layers, 512 neurons)
    ↓
Linear Decoder
    ↓
CTC Loss
    ↓
Transcribed Text
```

## Project Structure

```
├── data_preparation.py       # TIMIT dataset processing
├── feature_extraction.py     # Mel spectrogram, MFCC, PLP extraction
├── speechbrain_model.py      # ASR model architecture (ASRBrain class)
├── training_script.py        # Training pipeline with hyperparameters
├── simple_training.py        # Pre-trained model evaluation
├── realtime_asr_gui.py       # Tkinter GUI for transcription
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/GhofraneBenhassan8/Development-of-a-speech-recognition-system.git
cd Development-of-a-speech-recognition-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download TIMIT dataset:
   - Place TIMIT corpus in `data/TIMIT/` directory
   - Ensure structure: `TIMIT/TRAIN/` and `TIMIT/TEST/`

## Usage

### 1. Data Preparation
```bash
python data_preparation.py
```
This extracts SA1/SA2 sentences and creates CSV/JSON manifests.

### 2. Feature Extraction
```bash
python feature_extraction.py
```
Generates Mel spectrograms (80 bands) from audio files.

### 3. Model Training
```bash
python training_script.py
```
Trains the CNN-BiLSTM model with CTC loss. Training stops early if validation WER doesn't improve for 5 epochs.

### 4. Evaluation (Pre-trained Model)
```bash
python simple_training.py
```
Evaluates the LibriSpeech pre-trained model on TIMIT (baseline: 22.82% WER).

### 5. Real-time Transcription GUI
```bash
python realtime_asr_gui.py
```
Launch the graphical interface for:
- Recording audio from microphone
- Uploading audio files
- Viewing transcriptions with timestamps

## Hyperparameters

| Parameter        | Value |
|-----------       |-------|
| Sample Rate      | 16,000 Hz |
| Mel Bands        | 80 |
| Batch Size       | 8 |
| CNN Layers       | 3 (256 channels each) |
| LSTM Layers      | 4 (512 neurons, bidirectional) |
| Tokenizer        | BPE (120 tokens) |
| Learning Rate    | 0.0003 |
| Optimizer        | Adam |
| LR Scheduler     | ReduceLROnPlateau (patience=2) |
| Early Stopping   | Patience=5 epochs |
| Max Epochs       | 50 |

## Dataset: TIMIT

**Texas Instruments / MIT Corpus**
- **Speakers**: 630 (462 train, 168 test)
- **Dialects**: 8 American English regions (DR1-DR8)
- **Focus**: SA1 and SA2 calibration sentences
  - SA1: "She had your dark suit in greasy wash water all year"
  - SA2: "Don't ask me to carry an oily rag like that"
- **Total Files**: 1,260 (924 train, 336 test)
- **Duration**: ~3 seconds per file
- **Format**: WAV 16kHz, 16-bit

### Why SA1/SA2?
- Known by all speakers → enables direct comparison
- Inter-speaker variability → tests model robustness
- Phonetically rich → covers diverse English phonemes

## Model Performance

### Custom Model (Fine-tuned on TIMIT)
- **WER**: 5.00%
- **CER**: 5.05%
- **Test Files**: 42
- **Convergence**: Epoch 6

### Pre-trained Baseline (LibriSpeech)
- **WER**: 22.82%
- **CER**: 5.59%
- **Test Files**: 336

### Improvement
- **78% reduction** in Word Error Rate
- **10% reduction** in Character Error Rate

## Technology Stack

**Frameworks:**
- PyTorch 2.8.0
- SpeechBrain 1.0.3

**Audio Processing:**
- Librosa 0.11.0
- SoundFile 0.13.1
- SoundDevice (real-time recording)

**Evaluation:**
- JiWER 4.0.0 (WER/CER metrics)

**Data Science:**
- Pandas 2.3.3
- NumPy 2.0.2

**GUI:**
- Tkinter (multiplatform compatibility)

## GUI Features

- **Threading**: Non-blocking UI during model loading and transcription
- **Audio Capture**: Streaming microphone input with SoundDevice
- **File Support**: WAV, MP3, FLAC formats
- **Visual Feedback**:
  - Model status indicator (loading → ready)
  - Recording state (red indicator)
  - Transcription progress
  - Results with timestamps

## Training Process

1. **Data Preprocessing**: Extract SA1/SA2 sentences, create manifests
2. **Feature Extraction**: Mel spectrograms with 80 bands
3. **Tokenization**: BPE with SentencePiece (120 subword tokens)
4. **Training**: 90/10 train/validation split
5. **Early Stopping**: Monitors validation WER (patience=5)
6. **Evaluation**: Final test on held-out TIMIT test set

## Components

### 1. Feature Extraction
- **Mel Spectrogram**: 80 bands mimicking human auditory perception
- **Window**: 512 samples (32ms at 16kHz)
- **Hop**: 160 samples (10ms overlap)

### 2. CNN Encoder
- 3 convolutional layers (256 channels)
- Kernel size: 3
- BatchNorm + ReLU + Dropout (30%)
- Extracts local acoustic patterns

### 3. BiLSTM Encoder
- 4 bidirectional LSTM layers
- 512 neurons per direction (1024 total)
- Dropout: 30% between layers
- Models temporal dependencies

### 4. CTC Loss
- Enables automatic audio-text alignment
- No need for frame-level annotations
- Handles repetitions and silences

### 5. BPE Tokenization
- 120 subword tokens
- Handles unknown words gracefully
- Example: "greasy" → ["gre", "asy"]

## Future Improvements

- Extend to full TIMIT dataset (all 10 sentences per speaker)
- Test on multiple English dialects
- Real-time streaming transcription
- Model compression for mobile deployment
- Multi-language support

## Contributors

- **Esra Benltaief**
- **Ghofrane Benhassan**

**Supervisor**: M. Zied Laachiri

**Academic Year**: 2025/2026

## License

This project is part of an academic assignment. Please respect academic integrity guidelines if you use this code.

## Acknowledgments

- TIMIT Corpus creators (Texas Instruments & MIT)
- SpeechBrain team for the excellent framework
- LibriSpeech pre-trained models

## References

- SpeechBrain: https://speechbrain.github.io/
- TIMIT Corpus: https://catalog.ldc.upenn.edu/LDC93S1
- CTC Loss: Graves et al. (2006)
- Mel Spectrograms: Stevens & Volkmann (1940)

---

**For questions or issues, please open an issue on GitHub.**
