import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUGMENTATION = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08, fill_mode="constant"),
    tf.keras.layers.RandomZoom((-0.12, 0.12), fill_mode="constant"),
    tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
], name="geometric_augmentation")


def load_split(data_dir, train_per_class, augment_repeats):
    data_dir = Path(data_dir)
    grouped = {}

    for image_path in data_dir.rglob("*.png"):
        stem = image_path.stem
        if "_" not in stem:
            continue
        character, sample_text = stem.rsplit("_", 1)
        if not sample_text.isdigit():
            continue
        grouped.setdefault(character, []).append((int(sample_text), image_path))

    for character in grouped:
        grouped[character].sort(key=lambda item: item[0])

    characters = sorted(
        character
        for character, records in grouped.items()
        if len(records) > train_per_class
    )
    skipped_count = len(grouped) - len(characters)
    if skipped_count:
        print(f"Skipped {skipped_count} characters with at most {train_per_class} samples")
    if len(characters) == 0:
        raise RuntimeError(f"No characters with more than {train_per_class} samples found under {data_dir}")

    char_to_id = {character: index for index, character in enumerate(characters)}
    train_records = []
    test_records = []

    for character in characters:
        records = grouped[character]
        if len(records) <= train_per_class:
            raise RuntimeError(
                f"Character {character!r} has only {len(records)} samples; "
                f"need more than {train_per_class}"
            )
        class_id = char_to_id[character]
        train_records.extend((str(path), class_id) for _, path in records[:train_per_class])
        test_records.extend((str(path), class_id) for _, path in records[train_per_class:])

    train_records *= augment_repeats
    return characters, train_records, test_records


def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def augment(image, label):
    image = AUGMENTATION(image[tf.newaxis, ...], training=True)[0]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_dataset(records, batch_size, training):
    paths = np.asarray([item[0] for item in records])
    labels = np.asarray([item[1] for item in records], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(records), 10000), seed=SEED, reshuffle_each_iteration=True)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_mlp(num_classes):
    inputs = tf.keras.Input(shape=(50, 50, 1), name="handwriting_image")
    x = tf.keras.layers.Rescaling(1.0)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="character")(x)
    return tf.keras.Model(inputs, outputs, name="mlp")


def build_small_cnn(num_classes):
    inputs = tf.keras.Input(shape=(50, 50, 1), name="handwriting_image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="character")(x)
    return tf.keras.Model(inputs, outputs, name="small_cnn")


def build_deep_cnn(num_classes):
    inputs = tf.keras.Input(shape=(50, 50, 1), name="handwriting_image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="character")(x)
    return tf.keras.Model(inputs, outputs, name="deep_cnn")


def train_one(name, builder, train_ds, test_ds, num_classes, epochs, output_dir):
    model = builder(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    model_path = output_dir / f"{name}.keras"
    model.save(model_path)

    return {
        "name": name,
        "parameters": int(model.count_params()),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs_run": len(history.history["loss"]),
        "model_path": str(model_path),
    }


def main():
    parser = argparse.ArgumentParser(description="訓練繁體中文手寫辨識模型，可選 CPU 或 GPU")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--train-samples-per-class", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        print("Using CPU")
    elif args.device == "gpu":
        if not gpus:
            raise RuntimeError("No GPU detected; use --device cpu or --device auto")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", ", ".join(gpu.name for gpu in gpus))
    else:
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU:", ", ".join(gpu.name for gpu in gpus))
        else:
            print("Using CPU")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augment_repeats = max(1, (args.train_samples_per_class + args.train_per_class - 1) // args.train_per_class)
    characters, train_records, test_records = load_split(
        args.data_dir, args.train_per_class, augment_repeats
    )

    if args.smoke_test:
        characters = characters[:50]
        character_set = set(characters)
        train_records = [record for record in train_records if record[1] < 50]
        test_records = [record for record in test_records if record[1] < 50]

    num_classes = len(characters)
    train_ds = make_dataset(train_records, args.batch_size, training=True)
    test_ds = make_dataset(test_records, args.batch_size, training=False)

    results = []
    builders = {
        "mlp": build_mlp,
        "small_cnn": build_small_cnn,
        "deep_cnn": build_deep_cnn,
    }

    for name, builder in builders.items():
        result = train_one(name, builder, train_ds, test_ds, num_classes, args.epochs, output_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    results.sort(key=lambda item: item["test_accuracy"], reverse=True)
    with (output_dir / "model_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    with (output_dir / "model_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with (output_dir / "characters.json").open("w", encoding="utf-8") as file:
        json.dump(characters, file, ensure_ascii=False, indent=2)

    print("\nBest model:", results[0])


if __name__ == "__main__":
    main()
