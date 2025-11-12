import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import os
from pathlib import Path

class AudioPipeline:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Save outputs in the existing 'dataset' folder
        self.output_dir = r"C:\Users\PC\Documents\Formative2-Data-Preprocessing\dataset"

    # -------------------- VISUALIZATIONS --------------------  
    def load_and_visualize_audio(self, audio_path, save_viz=True):
        """Load audio and create waveform and spectrogram"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        filename = os.path.splitext(os.path.basename(audio_path))[0]

        if save_viz:
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)

            # Waveform
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(audio, sr=sr, alpha=0.8)
            plt.title(f'Waveform: {filename}', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{filename}_waveform.png'), dpi=300)
            plt.close()

            # Spectrogram
            plt.figure(figsize=(12, 6))
            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(f'Spectrogram: {filename}', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{filename}_spectrogram.png'), dpi=300)
            plt.close()

            print(f"  ✓ Visualizations saved for {filename}")
        return audio, sr

    # -------------------- AUGMENTATIONS --------------------
    def pitch_shift(self, audio, n_steps):
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

    def time_stretch(self, audio, rate):
        return librosa.effects.time_stretch(audio, rate=rate)

    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return np.clip(augmented, -1.0, 1.0)

    def add_background_noise(self, audio, noise_level=0.01):
        noise = np.random.randn(len(audio))
        freq = np.fft.rfftfreq(len(noise))
        freq[0] = 1
        pink_noise = np.fft.irfft(np.fft.rfft(noise) / np.sqrt(freq))

        if len(pink_noise) < len(audio):
            pink_noise = np.pad(pink_noise, (0, len(audio) - len(pink_noise)), mode='wrap')
        elif len(pink_noise) > len(audio):
            pink_noise = pink_noise[:len(audio)]

        pink_noise = pink_noise / np.max(np.abs(pink_noise))
        augmented = audio + noise_level * pink_noise
        return np.clip(augmented, -1.0, 1.0)

    def augment_audio(self, audio_path, output_dir):
        """Apply multiple augmentations to audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        augmentations = []

        # Pitch shift
        aug_audio = self.pitch_shift(audio, n_steps=2)
        aug_path = os.path.join(output_dir, f"{filename}_pitch_shift.wav")
        sf.write(aug_path, aug_audio, self.sample_rate)
        augmentations.append(aug_path)

        # Time stretch
        aug_audio = self.time_stretch(audio, rate=1.1)
        aug_path = os.path.join(output_dir, f"{filename}_time_stretch.wav")
        sf.write(aug_path, aug_audio, self.sample_rate)
        augmentations.append(aug_path)

        # Background noise
        aug_audio = self.add_background_noise(audio, noise_level=0.01)
        aug_path = os.path.join(output_dir, f"{filename}_noise.wav")
        sf.write(aug_path, aug_audio, self.sample_rate)
        augmentations.append(aug_path)

        return augmentations

    # -------------------- FEATURE EXTRACTION --------------------
    def extract_audio_features(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        features = {
            'file_name': os.path.basename(audio_path),
            'file_path': audio_path,
            'duration': len(audio) / sr
        }

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_energy_mean'] = np.mean(rms)
        features['rms_energy_std'] = np.std(rms)
        features['rms_energy_max'] = np.max(rms)
        features['rms_energy_min'] = np.min(rms)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        for i in range(12):
            features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        features['mel_spectrogram_mean'] = np.mean(mel_spec)
        features['mel_spectrogram_std'] = np.std(mel_spec)

        return features

    # -------------------- MAIN PIPELINE --------------------
    def process_all_audio(self, audio_input_dir, create_visualizations=True, create_augmentations=True):
        print("AUDIO PROCESSING PIPELINE")

        # Collect all audio files
        audio_files = []
        for root, dirs, files in os.walk(audio_input_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(root, file))

        print(f"\nFound {len(audio_files)} audio files")
        print(f"Input directory: {audio_input_dir}")
        print(f"Output directory: {self.output_dir}\n")

        all_audio_paths = []

        # Step 1: Visualizations
        if create_visualizations:
            print("\nCreating visualizations...")
            print("-" * 70)
            for audio_path in audio_files:
                print(f"Processing: {os.path.basename(audio_path)}")
                self.load_and_visualize_audio(audio_path, save_viz=True)
                all_audio_paths.append(audio_path)

        # Step 2: Augmentations
        augmented_dir = os.path.join(audio_input_dir, 'augmented')
        if create_augmentations:
            print("\nCreating augmentations (min 2 per sample)...")
            print("-" * 70)
            for audio_path in audio_files:
                print(f"Augmenting: {os.path.basename(audio_path)}")
                augmented_paths = self.augment_audio(audio_path, augmented_dir)
                all_audio_paths.extend(augmented_paths)
                print(f"  Created {len(augmented_paths)} augmented versions")

        # Step 3: Feature Extraction
        print("\nExtracting features...")
        all_features = []
        for audio_path in all_audio_paths:
            print(f"Extracting: {os.path.basename(audio_path)}")
            try:
                features = self.extract_audio_features(audio_path)

                # Metadata
                path_parts = Path(audio_path).parts
                filename = os.path.basename(audio_path).lower()

                features['member_id'] = 'unknown'
                for part in path_parts:
                    if 'member' in part.lower():
                        features['member_id'] = part
                        break

                if 'yes' in filename or 'approve' in filename:
                    features['phrase'] = 'yes_approve'
                elif 'confirm' in filename or 'transaction' in filename:
                    features['phrase'] = 'confirm_transaction'
                else:
                    features['phrase'] = 'unknown'

                if 'pitch' in filename:
                    features['augmentation'] = 'pitch_shift'
                elif 'time' in filename or 'stretch' in filename:
                    features['augmentation'] = 'time_stretch'
                elif 'noise' in filename:
                    features['augmentation'] = 'noise'
                else:
                    features['augmentation'] = 'original'

                all_features.append(features)
                print(f"  ✓ Extracted {len(features)} features")

            except Exception as e:
                print(f"  ✗ Error: {str(e)}")

        # Create and save DataFrame
        df = pd.DataFrame(all_features)
        priority_cols = ['file_name', 'member_id', 'phrase', 'augmentation', 'duration']
        other_cols = [col for col in df.columns if col not in priority_cols]
        df = df[priority_cols + other_cols]

        os.makedirs(self.output_dir, exist_ok=True)
        output_csv = os.path.join(self.output_dir, 'audio_features.csv')
        df.to_csv(output_csv, index=False)

        # Summary
        print("\nPIPELINE COMPLETE")
        print(f"Total audio samples processed: {len(all_audio_paths)}")
        print(f"Features extracted: {len(df.columns)}")
        print(f"Output saved to: {output_csv}")
        print("\nDataset Summary:")
        print(f"  Members: {df['member_id'].nunique()}")
        print(f"  Phrases: {df['phrase'].nunique()}")
        print(f"  Total samples: {len(df)}")
        print(f"  Original samples: {len(df[df['augmentation'] == 'original'])}")
        print(f"  Augmented samples: {len(df[df['augmentation'] != 'original'])}")

        print("\nSample distribution:")
        print(df['member_id'].value_counts().to_string())

        return df

# -------------------- MAIN FUNCTION --------------------
def main():
    audio_input_dir = r"C:\Users\PC\Documents\Formative2-Data-Preprocessing\dataset\audios"
    
    if not os.path.exists(audio_input_dir):
        print("ERROR: Audio directory not found!")
        print(f"Please update 'audio_input_dir' to point to your recorded audio files")
        return

    pipeline = AudioPipeline(sample_rate=16000)

    df = pipeline.process_all_audio(
        audio_input_dir=audio_input_dir,
        create_visualizations=True,
        create_augmentations=True
    )

    print("\nFirst 5 rows of audio_features.csv:")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
