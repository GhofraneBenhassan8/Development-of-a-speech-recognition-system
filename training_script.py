"""
Script d'entraînement du modèle ASR sur TIMIT SA1/SA2 avec BPE (SentencePiece),
régularisation, optimisation et early stopping custom.
"""
import os
import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
import pandas as pd
import sentencepiece as spm


# --- Early stopping custom ---
class EarlyStopper:
    def __init__(self, patience=3, min_key="WER"):
        self.patience = patience
        self.min_key = min_key
        self.best_score = None
        self.counter = 0

    def step(self, current_score):
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
            return False  # continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # stop training
            return False


def dataio_prepare(hparams):
    """Prépare les données pour l'entraînement"""
    train_data = pd.read_csv(hparams["train_csv"])
    valid_data = pd.read_csv(hparams["valid_csv"])
    test_data = pd.read_csv(hparams["test_csv"])

    tokenizer = hparams["tokenizer"]

    datasets = {}
    for split, df in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        data_dict = {}
        for idx, row in df.iterrows():
            data_dict[row['speaker_id'] + '_' + row['sentence_id']] = {
                "wav": row['audio_path'],
                "transcript": row['transcription'],
                "duration": 1.0,
            }

        dataset = DynamicItemDataset(data_dict)

        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            sig = sb.dataio.dataio.read_audio(wav)
            return sig

        @sb.utils.data_pipeline.takes("transcript")
        @sb.utils.data_pipeline.provides("tokens")
        def text_pipeline(transcript):
            tokens_list = tokenizer.encode(transcript.lower())
            return torch.LongTensor(tokens_list)

        dataset.add_dynamic_item(audio_pipeline)
        dataset.add_dynamic_item(text_pipeline)

        dataset.set_output_keys(["id", "sig", "tokens"])
        datasets[split] = dataset

    return datasets


def create_hparams():
    """Crée les hyperparamètres pour l'entraînement"""
    hparams = {
        "train_csv": "../data/manifests/train_sa.csv",
        "valid_csv": "../data/manifests/valid_sa.csv",
        "test_csv": "../data/manifests/test_sa.csv",

        "output_folder": "../results/timit_sa_bpe",   # ✅ nouveau dossier pour éviter conflits
        "save_folder": "../results/timit_sa_bpe/save",

        "sample_rate": 16000,
        "n_mels": 80,

        "n_cnn_layers": 3,
        "cnn_channels": 256,
        "cnn_kernelsize": 3,
        "rnn_layers": 4,
        "rnn_neurons": 512,
        "rnn_bidirectional": True,
        "dnn_neurons": 512,

        "encoder_dim": 512,
        "decoder_dim": 512,
        "num_layers": 4,

        "number_of_epochs": 50,
        "batch_size": 8,
        "lr": 0.0003,
        "sorting": "ascending",

        "seed": 1234,
    }
    return hparams


def train_model():
    """Fonction principale d'entraînement"""
    hparams = create_hparams()

    os.makedirs(hparams["output_folder"], exist_ok=True)
    os.makedirs(hparams["save_folder"], exist_ok=True)

    torch.manual_seed(hparams["seed"])

    print("Préparation des données...")

    train_df = pd.read_csv(hparams["train_csv"])
    valid_size = int(len(train_df) * 0.1)
    valid_df = train_df.sample(n=valid_size, random_state=hparams["seed"])
    train_df = train_df.drop(valid_df.index)

    train_df.to_csv(hparams["train_csv"].replace("train_sa", "train_sa_split"), index=False)
    valid_df.to_csv(hparams["valid_csv"], index=False)

    print(f"Train: {len(train_df)} samples")
    print(f"Valid: {len(valid_df)} samples")

    # --- BPE Tokenizer avec SentencePiece ---
    all_transcripts = [t.lower() for t in train_df['transcription'].tolist()]
    text_file = os.path.join(hparams["output_folder"], "train_text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for line in all_transcripts:
            f.write(line + "\n")

    spm.SentencePieceTrainer.Train(
        input=text_file,
        model_prefix=os.path.join(hparams["output_folder"], "bpe"),
        vocab_size=120,   # ajusté pour ton corpus
        model_type="bpe"
    )

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(os.path.join(hparams["output_folder"], "bpe.model"))

    hparams["tokenizer"] = tokenizer
    hparams["output_neurons"] = tokenizer.get_piece_size()
    hparams["blank_index"] = 0

    print(f"Vocabulaire BPE: {hparams['output_neurons']} tokens")

    datasets = dataio_prepare(hparams)

    from speechbrain_model import ASRBrain, build_model
    modules = build_model(hparams)

    from speechbrain.utils.epoch_loop import EpochCounter
    hparams["epoch_counter"] = EpochCounter(limit=hparams["number_of_epochs"])

    from speechbrain.nnet.losses import ctc_loss
    from speechbrain.utils.metric_stats import ErrorRateStats
    hparams["ctc_cost"] = ctc_loss
    hparams["log_softmax"] = torch.nn.LogSoftmax(dim=-1)
    hparams["error_rate_computer"] = lambda: ErrorRateStats()
    hparams["cer_computer"] = lambda: ErrorRateStats(split_tokens=True)

    from speechbrain.utils.checkpoints import Checkpointer
    hparams["checkpointer"] = Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables={"model": modules}
    )

    # --- Optimisation ---
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    opt_class = lambda params: torch.optim.Adam(params, lr=hparams["lr"], weight_decay=1e-5)
    hparams["lr_scheduler_class"] = ReduceLROnPlateau

    # --- Early stopping custom ---
    hparams["early_stopper"] = EarlyStopper(patience=5, min_key="WER")

    asr_brain = ASRBrain(
        modules=modules,
        opt_class=opt_class,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    # ✅ Injecter lr_scheduler_class comme attribut du namespace
    asr_brain.hparams.lr_scheduler_class = hparams["lr_scheduler_class"]

    print("\nDébut de l'entraînement...")
    asr_brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs={"batch_size": hparams["batch_size"]},
        valid_loader_kwargs={"batch_size": hparams["batch_size"]},
    )

    print("\nÉvaluation sur le test set...")
    asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs={"batch_size": hparams["batch_size"]},
    )

    print(f"\nEntraînement terminé! Modèle sauvegardé dans {hparams['save_folder']}")


if __name__ == "__main__":
    print("=" * 60)
    print("ENTRAÎNEMENT ASR - TIMIT SA1/SA2")
    print("=" * 60)
    train_model()