# prepare_train.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_DIR = "dataset"
MODEL_DIR = "models"
ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
ensure_dir(MODEL_DIR)

def load_samples_from_disk(data_dir=DATA_DIR):
    X = []
    y = []
    filepaths = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir): continue
        for fname in os.listdir(label_dir):
            if not fname.endswith(".json"): continue
            with open(os.path.join(label_dir, fname), "r") as f:
                j = json.load(f)
            # use relative landmarks (translation-invariant)
            rel = j.get("relative_landmarks", None)
            if rel is None:
                rel = j.get("landmarks")
                # fallback - compute relative in code if necessary
            if rel is None:
                continue
            flat = np.array(rel).flatten()  # 21*3 = 63
            if np.isnan(flat).any():
                continue
            X.append(flat)
            y.append(j["label"])
            filepaths.append(os.path.join(label_dir, fname))
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} samples from {data_dir}")
    return X, y

# --- load
X, y = load_samples_from_disk()

# --- label encode
le = LabelEncoder()
y_enc = le.fit_transform(y)
label_map = {int(idx): label for idx, label in enumerate(le.classes_)}
# save label map
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({"classes": list(le.classes_)}, f)

# --- split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# --- scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# --- RandomForest baseline
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
pred = rf.predict(X_test_s)
print("RF accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_hand_sign.pkl"))

# --- OPTIONAL: small Keras MLP (if tensorflow installed)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, utils, models
    num_classes = len(le.classes_)
    y_train_cat = utils.to_categorical(y_train, num_classes)
    y_test_cat = utils.to_categorical(y_test, num_classes)
    model = models.Sequential([
        layers.Input(shape=(X_train_s.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train_cat, epochs=25, batch_size=32, validation_split=0.1)
    loss, acc = model.evaluate(X_test_s, y_test_cat)
    print("Keras test acc:", acc)
    model.save(os.path.join(MODEL_DIR, "keras_hand_sign.h5"))
except Exception as e:
    print("TensorFlow not installed or training failed:", e)
