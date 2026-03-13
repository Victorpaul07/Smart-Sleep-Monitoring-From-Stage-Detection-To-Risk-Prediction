import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from joblib import dump

DATA_DIR = "data/sleep-edf"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def find_hypnogram(psg_name, file_list):
    """Find matching hypnogram using your logic"""
    subject = psg_name.split('-')[0]
    base = subject[:-1]  # remove last char (0/E)
    for f in file_list:
        if f.startswith(base) and f.endswith("Hypnogram.edf"):
            return f
    return None


def load_data(dataset_path):
    """Your exact loading logic + improvements"""
    X, y = [], []
    files = sorted(os.listdir(dataset_path))
    psg_files = [f for f in files if f.endswith("-PSG.edf")]

    print(f"Found {len(psg_files)} PSG files")

    for psg_file in psg_files:
        hyp_file = find_hypnogram(psg_file, files)
        if hyp_file is None:
            print(f"Skipping {psg_file} (no hypnogram)")
            continue

        print(f"Loading: {psg_file} -> {hyp_file}")

        try:
            # Read PSG
            raw = mne.io.read_raw_edf(
                os.path.join(dataset_path, psg_file),
                preload=True, verbose=False
            )

            # Resample to 100Hz (standard)
            raw.resample(100)

            # Read + apply annotations
            annotations = mne.read_annotations(
                os.path.join(dataset_path, hyp_file)
            )
            raw.set_annotations(annotations)

            # Extract events (sleep stages)
            events, event_id = mne.events_from_annotations(raw)
            print(f"  Events: {len(events)}, classes: {list(event_id.keys())}")

            # Create 30s epochs
            epochs = mne.Epochs(
                raw,
                events,
                event_id,
                tmin=0,
                tmax=30 - 1 / raw.info['sfreq'],
                baseline=None,
                preload=True,
                verbose=False
            )

            data = epochs.get_data()  # (n_epochs, n_channels, 3000)
            labels_raw = epochs.events[:, -1]

            id_to_stage = {v: k for k, v in epochs.event_id.items()}

            # Map to 0-4 (your logic)
            for i, label in enumerate(labels_raw):
                stage = id_to_stage[label]
                if stage == "Sleep stage W":
                    y.append(0)
                    X.append(data[i])
                elif stage == "Sleep stage 1":
                    y.append(1)
                    X.append(data[i])
                elif stage == "Sleep stage 2":
                    y.append(2)
                    X.append(data[i])
                elif stage in ["Sleep stage 3", "Sleep stage 4"]:
                    y.append(3)
                    X.append(data[i])
                elif stage == "Sleep stage R":
                    y.append(4)
                    X.append(data[i])
                # Skip unknown (Movement/Unknown)

            print(f"  Added {len([l for l in y if l is not None])} epochs")

        except Exception as e:
            print(f"  Error loading {psg_file}: {e}")
            continue

    if not X:
        raise RuntimeError("No valid epochs loaded!")

    X = np.array(X)  # (n_epochs, n_channels, time)
    y = np.array(y)  # (n_epochs,)

    # Use first EEG channel (usually index 0 or pick EEG)
    eeg_idx = 0  # Fpz-Cz typically first
    # Or auto-pick: eeg_idx = mne.pick_types(raw.info, eeg=True)[0]
    X = X[:, eeg_idx, :]  # (n_epochs, 3000)

    print(f"TOTAL: {X.shape[0]} epochs")
    return X, y


def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(32, 7, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')  # 5 classes
    ])
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("Loading Sleep-EDF data...")
    X, y = load_data(DATA_DIR)
    print(f"Loaded {len(y)} epochs, classes: {np.unique(y)}")

    # Normalize per epoch (z-score)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    X = X[..., np.newaxis]  # Add channel dim: (epochs, 3000, 1)

    # Labels are already 0=W,1=N1,2=N2,3=N3,4=REM
    print("Labels range:", y.min(), "to", y.max())
    y_enc = y  # Already encoded!
    y_cat = to_categorical(y_enc, 5)

    # Save encoder for Flask predictions
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['W', 'N1', 'N2', 'N3', 'REM'])
    dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Build & train
    model = build_cnn_lstm((X.shape[1], 1))

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'sleep_cnn_lstm_best.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, batch_size=64,  # Larger batch for 21k samples
        callbacks=[checkpoint],
        verbose=1
    )

    # Save final model
    model.save(os.path.join(MODEL_DIR, 'sleep_cnn_lstm_final.h5'))
    print("Training complete! Models saved.")

if __name__ == "__main__":
    main()
