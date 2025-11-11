"""
train_waste_classifier.py — Train & fine-tune MobileNetV2 for household waste classification
-----------------------------------------------------------------------------------
Dataset structure:
    data/
      ├── battery/
      ├── biological/
      ├── cardboard/
      ├── clothes/
      ├── glass/
      ├── metal/
      ├── paper/
      ├── plastic/
      ├── shoes/
      └── trash/
"""

import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==== CONFIG ====
DATA_DIR   = r"C:\Users\Dell\Downloads\waste-classifier\data"
IMG_SIZE   = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FT_EPOCHS   = 15
SEED        = 42

HEAD_LR = 5e-4
FT_LR   = 1e-5

# ==== LOAD DATA ====
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("[INFO] Classes:", classes)

image_paths, labels = [], []
for c in classes:
    cdir = os.path.join(DATA_DIR, c)
    for f in os.listdir(cdir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(cdir, f))
            labels.append(c)
df = pd.DataFrame({"filename": image_paths, "class": labels})
print(f"[INFO] {len(df)} images found.")

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["class"], random_state=SEED)
valid_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["class"], random_state=SEED)

# ==== CLASS WEIGHTS ====
y_train = train_df['class'].map({c:i for i,c in enumerate(sorted(train_df['class'].unique()))})
class_weights_raw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights_raw)}

# ==== IMAGE GENERATORS ====
train_gen = ImageDataGenerator(
    rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, shear_range=0.1, horizontal_flip=True, fill_mode='nearest'
).flow_from_dataframe(
    train_df, x_col="filename", y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE), class_mode="categorical", batch_size=BATCH_SIZE
)

valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    valid_df, x_col="filename", y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE), class_mode="categorical", batch_size=BATCH_SIZE
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, x_col="filename", y_col="class",
    target_size=(IMG_SIZE, IMG_SIZE), class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False
)

num_classes = len(train_gen.class_indices)
print("[INFO] Detected classes:", list(train_gen.class_indices.keys()))

# ==== BUILD MODEL ====
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
base.trainable = False
x = layers.Dropout(0.3)(base.output)
out = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(optimizer=optimizers.Adam(learning_rate=HEAD_LR), loss="categorical_crossentropy", metrics=["accuracy"])
print("[INFO] Training classification head...")
model.fit(train_gen, validation_data=valid_gen, epochs=HEAD_EPOCHS, class_weight=class_weights, verbose=1)

print("[INFO] Fine-tuning full model...")
base.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=FT_LR), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=valid_gen, epochs=FT_EPOCHS, class_weight=class_weights, verbose=1)

# ==== SAVE MODEL ====
os.makedirs("runs", exist_ok=True)
model.save("runs/mobilenetv2_waste_model_finetuned.h5")
print("\n✅ [INFO] Model saved successfully at: runs/mobilenetv2_waste_model_finetuned.h5")
