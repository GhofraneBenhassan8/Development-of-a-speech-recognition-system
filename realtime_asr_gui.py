"""
Interface graphique pour transcription audio
"""
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
from pathlib import Path
import torch
from speechbrain.inference.ASR import EncoderDecoderASR
from datetime import datetime
import os


class SimpleASRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 ASR TIMIT")
        self.root.geometry("700x500")
        
        self.is_recording = False
        self.recorded_audio = None
        self.asr_model = None
        self.current_file = None
        self.RATE = 16000
        
        self.create_widgets()
        self.load_model_thread()
    
    def create_widgets(self):
        # Status
        self.status_label = tk.Label(
            self.root,
            text="📦 Chargement du modèle...",
            font=('Arial', 11),
            fg='orange'
        )
        self.status_label.pack(pady=10)
        
        # Enregistrement
        frame1 = tk.LabelFrame(self.root, text="🎙️ Enregistrement", padx=10, pady=10)
        frame1.pack(fill=tk.X, padx=20, pady=5)
        
        self.rec_btn = tk.Button(
            frame1,
            text="● Enregistrer",
            font=('Arial', 12),
            bg='green',
            fg='white',
            command=self.toggle_record,
            state=tk.DISABLED,
            width=15
        )
        self.rec_btn.pack(side=tk.LEFT, padx=5)
        
        self.trans_rec_btn = tk.Button(
            frame1,
            text="📝 Transcrire",
            font=('Arial', 12),
            bg='blue',
            fg='white',
            command=self.transcribe_recording,
            state=tk.DISABLED,
            width=15
        )
        self.trans_rec_btn.pack(side=tk.LEFT, padx=5)
        
        self.rec_status = tk.Label(frame1, text="", fg='red', font=('Arial', 10, 'bold'))
        self.rec_status.pack(pady=5)
        
        # Fichier
        frame2 = tk.LabelFrame(self.root, text="📁 Fichier", padx=10, pady=10)
        frame2.pack(fill=tk.X, padx=20, pady=5)
        
        self.browse_btn = tk.Button(
            frame2,
            text="📂 Parcourir",
            font=('Arial', 11),
            command=self.browse,
            state=tk.DISABLED,
            width=15
        )
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.trans_file_btn = tk.Button(
            frame2,
            text="📝 Transcrire fichier",
            font=('Arial', 11),
            bg='blue',
            fg='white',
            command=self.transcribe_file,
            state=tk.DISABLED,
            width=18
        )
        self.trans_file_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(frame2, text="", fg='green', font=('Arial', 9))
        self.file_label.pack(pady=5)
        
        # Transcription
        frame3 = tk.LabelFrame(self.root, text="📄 Résultats", padx=10, pady=10)
        frame3.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        self.text = scrolledtext.ScrolledText(frame3, font=('Courier', 10), wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True)
        
        # Effacer
        tk.Button(
            self.root,
            text="🗑️ Effacer",
            command=lambda: self.text.delete(1.0, tk.END),
            bg='red',
            fg='white'
        ).pack(pady=5)
        
        # Info
        self.info = tk.Label(self.root, text="", font=('Arial', 9), fg='gray')
        self.info.pack(pady=5)
    
    def load_model_thread(self):
        threading.Thread(target=self.load_model, daemon=True).start()
    
    def load_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-crdnn-rnnlm-librispeech",
                savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
                run_opts={"device": device}
            )
            self.root.after(0, self.on_loaded)
        except Exception as err:
            error_msg = str(err)
            self.root.after(0, lambda: messagebox.showerror("Erreur", error_msg))
    
    def on_loaded(self):
        self.status_label.config(text="✅ Modèle chargé!", fg='green')
        self.rec_btn.config(state=tk.NORMAL)
        self.browse_btn.config(state=tk.NORMAL)
        self.info.config(text="Prêt")
    
    def toggle_record(self):
        if not self.is_recording:
            self.start_record()
        else:
            self.stop_record()
    
    def start_record(self):
        self.is_recording = True
        self.recording_data = []
        self.rec_btn.config(text="■ Arrêter", bg='red')
        self.rec_status.config(text="● ENREGISTREMENT")
        self.trans_rec_btn.config(state=tk.DISABLED)
        
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.RATE,
            callback=lambda d, f, t, s: self.recording_data.append(d.copy()) if self.is_recording else None
        )
        self.stream.start()
    
    def stop_record(self):
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.recording_data:
            self.recorded_audio = np.concatenate(self.recording_data, axis=0)
        
        self.rec_btn.config(text="● Enregistrer", bg='green')
        self.rec_status.config(text="")
        
        if self.recorded_audio is not None:
            self.trans_rec_btn.config(state=tk.NORMAL)
            self.info.config(text="✓ Enregistrement prêt")
    
    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3 *.flac"), ("Tous", "*.*")]
        )
        if path:
            self.current_file = path
            self.file_label.config(text=f"✓ {Path(path).name}")
            self.trans_file_btn.config(state=tk.NORMAL)
    
    def transcribe_recording(self):
        if self.recorded_audio is None:
            return
        
        self.trans_rec_btn.config(state=tk.DISABLED)
        self.info.config(text="⏳ Transcription...")
        
        threading.Thread(
            target=self._transcribe_audio,
            args=(self.recorded_audio, "Enregistrement"),
            daemon=True
        ).start()
    
    def transcribe_file(self):
        if not self.current_file:
            return
        
        self.trans_file_btn.config(state=tk.DISABLED)
        self.info.config(text="⏳ Transcription...")
        
        threading.Thread(
            target=self._transcribe_file,
            args=(self.current_file,),
            daemon=True
        ).start()
    
    def _transcribe_audio(self, audio, name):
        try:
            print(f"DEBUG: Début transcription {name}")
            print(f"DEBUG: Audio shape: {audio.shape}")
            
            # Utiliser le dossier actuel
            temp_file = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            print(f"DEBUG: Fichier temp: {temp_file}")
            
            # Sauver
            sf.write(temp_file, audio, self.RATE)
            print(f"DEBUG: Fichier sauvé")
            
            # Vérifier que le fichier existe
            if not os.path.exists(temp_file):
                raise Exception(f"Fichier non créé: {temp_file}")
            
            print(f"DEBUG: Transcription...")
            # Transcrire
            result = self.asr_model.transcribe_file(temp_file)
            print(f"DEBUG: Résultat: {result}")
            
            # Nettoyer
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Afficher
            time = datetime.now().strftime("%H:%M:%S")
            text = f"[{time}] {name}:\n{result}\n\n"
            self.root.after(0, lambda t=text: self._show_result(t))
            
        except Exception as err:
            print(f"DEBUG: ERREUR: {err}")
            import traceback
            traceback.print_exc()
            error_msg = str(err)
            self.root.after(0, lambda msg=error_msg: self._show_error(msg))
    
    def _transcribe_file(self, filepath):
        try:
            print(f"DEBUG: Fichier original: {filepath}")
            
            # Normaliser le chemin
            filepath = os.path.normpath(filepath)
            print(f"DEBUG: Fichier normalisé: {filepath}")
            
            # Vérifier existence
            if not os.path.exists(filepath):
                raise Exception(f"Fichier introuvable: {filepath}")
            
            print(f"DEBUG: Transcription du fichier...")
            # Transcrire directement
            result = self.asr_model.transcribe_file(filepath)
            print(f"DEBUG: Résultat: {result}")
            
            # Afficher
            time = datetime.now().strftime("%H:%M:%S")
            name = Path(filepath).name
            text = f"[{time}] {name}:\n{result}\n\n"
            self.root.after(0, lambda t=text: self._show_result(t, True))
            
        except Exception as err:
            print(f"DEBUG: ERREUR fichier: {err}")
            import traceback
            traceback.print_exc()
            error_msg = str(err)
            self.root.after(0, lambda msg=error_msg: self._show_error(msg, True))
    
    def _show_result(self, text, is_file=False):
        self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.info.config(text="✓ Terminé")
        
        if is_file:
            self.trans_file_btn.config(state=tk.NORMAL)
        else:
            self.rec_btn.config(state=tk.NORMAL)
    
    def _show_error(self, msg, is_file=False):
        messagebox.showerror("Erreur", msg)
        self.info.config(text="✗ Erreur")
        
        if is_file:
            self.trans_file_btn.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = SimpleASRGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()