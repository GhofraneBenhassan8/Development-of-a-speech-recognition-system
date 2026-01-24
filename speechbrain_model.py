"""
Modèle de reconnaissance vocale avec SpeechBrain + SentencePiece BPE
"""
import torch
import torch.nn as nn
import speechbrain as sb
import torchaudio


class ASRBrain(sb.Brain):
    """Classe principale pour l'entraînement ASR avec SpeechBrain"""

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        feats = self.modules.compute_features(wavs)
        feats = feats.transpose(1, 2)
        feats = self.modules.normalize(feats, wav_lens)
        feats = feats.transpose(1, 2)

        encoded = self.modules.encoder(feats)
        encoded = encoded.transpose(1, 2)
        encoded, _ = self.modules.encoder_rnn(encoded)

        logits = self.modules.decoder(encoded)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        p_ctc, wav_lens = predictions
        ids = batch.id

        tokens = batch.tokens.data
        token_lens = batch.tokens.lengths

        # CTC Loss
        loss = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, token_lens, self.hparams.blank_index
        )

        if stage != sb.Stage.TRAIN:
            # Greedy decode -> indices
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

            # Convertir indices -> sous-mots avec SentencePiece
            pred_words = [self.hparams.tokenizer.decode_ids(seq) for seq in sequence]
            target_words = [self.hparams.tokenizer.decode_ids(t.tolist()) for t in tokens]

            # Calcul WER / CER
            self.wer_metric.append(ids, pred_words, target_words)
            self.cer_metric.append(ids, pred_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {
                "loss": stage_loss,
                "CER": self.cer_metric.summarize("error_rate"),
                "WER": self.wer_metric.summarize("error_rate"),
            }

            if stage == sb.Stage.VALID:
                print(f"Epoch {epoch}: Valid loss={stats['loss']:.4f}, "
                      f"WER={stats['WER']:.2f}%, CER={stats['CER']:.2f}%")

                # ✅ Correct usage of ReduceLROnPlateau
                if not hasattr(self, "lr_scheduler"):
                    self.lr_scheduler = self.hparams.lr_scheduler_class(
                        self.optimizer, mode="min", patience=2, factor=0.5
                    )

                self.lr_scheduler.step(stats["loss"])
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Learning rate adjusted to {current_lr:.6f}")

                # ✅ Early stopping check
                if self.hparams.early_stopper.step(stats["WER"]):
                    print("Early stopping triggered.")
                    # Forcer la fin de l'entraînement
                    self.hparams.epoch_counter.current = self.hparams.epoch_counter.limit

                # Sauvegarde checkpoint
                self.checkpointer.save_and_keep_only(
                    meta=stats, min_keys=["WER"]
                )

            elif stage == sb.Stage.TEST:
                print(f"Test - WER: {stats['WER']:.2f}%, CER: {stats['CER']:.2f}%")


class MelSpectrogramExtractor(nn.Module):
    """Wrapper for mel spectrogram extraction"""
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels
        )

    def forward(self, wavs):
        feats = self.mel_spec(wavs)
        feats = torch.log(feats + 1e-9)
        return feats


def build_model(hparams):
    modules = nn.ModuleDict()

    modules["compute_features"] = MelSpectrogramExtractor(
        sample_rate=16000,
        n_mels=hparams["n_mels"]
    )

    modules["normalize"] = sb.processing.features.InputNormalization()

    class CNNEncoder(nn.Module):
        def __init__(self, n_mels, n_layers, channels, kernel_size, dropout=0.3):
            super().__init__()
            layers = []
            in_channels = n_mels
            for i in range(n_layers):
                layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=1,
                        bias=False
                    )
                )
                layers.append(nn.BatchNorm1d(channels))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_channels = channels
            self.cnn = nn.Sequential(*layers)

        def forward(self, x):
            return self.cnn(x)

    modules["encoder"] = CNNEncoder(
        n_mels=hparams["n_mels"],
        n_layers=hparams["n_cnn_layers"],
        channels=hparams["cnn_channels"],
        kernel_size=hparams["cnn_kernelsize"],
        dropout=0.3
    )

    modules["encoder_rnn"] = nn.LSTM(
        input_size=hparams["cnn_channels"],
        hidden_size=hparams["rnn_neurons"],
        num_layers=hparams["rnn_layers"],
        bidirectional=hparams["rnn_bidirectional"],
        batch_first=True,
        dropout=0.3 if hparams["rnn_layers"] > 1 else 0,
    )

    rnn_output_size = hparams["rnn_neurons"] * 2 if hparams["rnn_bidirectional"] else hparams["rnn_neurons"]
    modules["decoder"] = nn.Linear(
        rnn_output_size,
        hparams["output_neurons"]
    )

    return modules