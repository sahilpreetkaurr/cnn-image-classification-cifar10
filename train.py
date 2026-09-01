"""
Image Classification Using CNN on CIFAR-10
E&ICT Academy, IIT Roorkee

Run:
    python train.py
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
BATCH_SIZE = 64
EPOCHS = 25

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train_full = y_train_full.squeeze()
    y_test = y_test.squeeze()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.10,
        random_state=SEED,
        stratify=y_train_full,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.50)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cifar10_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    ax.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history["loss"], label="Training Loss")
    ax.plot(epochs, history.history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curve.png", dpi=200)
    plt.close(fig)


def evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("CIFAR-10 Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    return predictions


def plot_sample_predictions(x_test, y_test, predictions, n=12):
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(x_test), size=n, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(x_test[idx])
        true_name = CLASS_NAMES[y_test[idx]]
        pred_name = CLASS_NAMES[predictions[idx]]
        ax.set_title(f"True: {true_name}\nPred: {pred_name}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions.png", dpi=200)
    plt.close(fig)


def main():
    set_seed()

    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    x_train, x_val, x_test, y_train, y_val, y_test = load_data()

    print("\nDataset shapes")
    print("Training:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)
    predictions = evaluate_model(model, x_test, y_test)
    plot_sample_predictions(x_test, y_test, predictions)

    model.save(MODELS_DIR / "cifar10_cnn.keras")
    print("\nSaved model and figures successfully.")


if __name__ == "__main__":
    main()
