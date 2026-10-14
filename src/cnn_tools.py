from __future__ import annotations

from pathlib import Path
from time import perf_counter
import numpy as np


def build_cnn(input_shape=(224, 224, 3)):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(192, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="farmtech_cnn_from_scratch",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def make_tf_dataset(image_paths, labels, image_size=(224, 224), batch_size=8, training=False, seed=42):
    import tensorflow as tf

    paths = np.asarray([str(p) for p in image_paths])
    labels = np.asarray(labels, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image.set_shape((*image_size, 3))
        return image, label

    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def paths_and_labels(dataset_root: str | Path, split: str):
    root = Path(dataset_root) / "images" / split
    paths, labels = [], []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if p.stem.lower().startswith("person_"):
            label = 0
        elif p.stem.lower().startswith("car_"):
            label = 1
        else:
            continue
        paths.append(p)
        labels.append(label)
    return paths, labels


def measure_keras_inference(model, dataset) -> float:
    timings = []
    for images, _ in dataset.unbatch().batch(1):
        start = perf_counter()
        _ = model(images, training=False)
        timings.append((perf_counter() - start) * 1000)
    return float(np.mean(timings)) if timings else 0.0
