import os
import time
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

# =========================
#       MODEL CREATION
# =========================
def create_model(input_shape=(224, 224, 3), l2_factor=0.0001, dropout_rate=0.3):
    """
    Create a VGG16 transfer learning model with a custom classifier head.
    """
    base_model = tf.keras.applications.VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2_factor))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_factor))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


# =========================
#       TRAINING
# =========================
def train_model(model, train_generator, validation_generator, epochs, model_path, fine_tune=False, callbacks=None):
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)

    all_callbacks = [early_stop, checkpoint]
    if callbacks:
        all_callbacks.extend(callbacks)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=all_callbacks
    )

    if fine_tune:
        print("Starting fine-tuning...")

        base = model.layers[1] if isinstance(model.layers[1], tf.keras.Model) else None
        if base:
            base.trainable = True
            for layer in base.layers[:-4]:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )

        fine_history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=max(1, epochs // 2),
            callbacks=all_callbacks
        )

        for key in history.history.keys():
            history.history[key] += fine_history.history[key]

    return history


# =========================
#       SAVE and LOAD
# =========================
def save_model(model, path):
    if not path.endswith(".keras"):
        path += ".keras"
    model.save(path)
    print("Model saved at", path)
    return path


def load_trained_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if not (path.endswith(".keras") or path.endswith(".h5")):
        raise ValueError(
            "Keras 3 only supports V3 `.keras` files or legacy `.h5` files. "
            f"Given: {path}"
        )

    model = tf.keras.models.load_model(path)
    print("Loaded model from", path)
    return model


# =========================
#       RETRAINING
# =========================
def retrain_model(
    new_data_folder,
    original_train_folder,
    batch_size,
    epochs,
    output_directory,
    fine_tune=True,
    callbacks=None
):
    print("Starting retraining process")

    # Merge new images into existing training set
    for root, dirs, files in os.walk(new_data_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                source = os.path.join(root, file)
                label = os.path.basename(root)
                target_folder = os.path.join(original_train_folder, label)
                os.makedirs(target_folder, exist_ok=True)
                shutil.copy(source, os.path.join(target_folder, file))

    print("New data merged into training folder")

    # Image generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        original_train_folder,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        original_train_folder,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    # Load latest model if exists
    model_files = [f for f in os.listdir(output_directory) if f.endswith(".keras")]
    if model_files:
        latest = max(model_files)
        model_path = os.path.join(output_directory, latest)
        model = load_trained_model(model_path)
    else:
        print("No previous model found. Creating a new one")
        model = create_model()

    # Training and versioning
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_directory, f"model_{timestamp}.keras")

    history = train_model(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator,
        epochs=epochs,
        model_path=save_path,
        fine_tune=fine_tune,
        callbacks=callbacks  # <-- pass callbacks here
    )

    print("Retraining complete")
    print("New model saved as", save_path)

    return save_path, history
