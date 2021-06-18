import json
import random
import threading
import time
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

label_to_number_dict = None
number_to_label_dict = None


def load_dictionaries():
    global label_to_number_dict
    global number_to_label_dict
    with open("../Organizing/labels/label_to_number_dict.json", "r") as f:
        label_to_number_dict = json.load(f)
    with open("../Organizing/labels/number_to_label_dict.json", "r") as f:
        number_to_label_dict = json.load(f)


def get_current_train_pack_of_data(train_pack_number, size_string):
    # If preloaded, just return the preloaded data
    if PRELOAD_DATA and preloaded_train_feat is not None:
        return preloaded_train_feat, preloaded_train_labels

    # Else load it from the disk
    loaded_training_labels = np.load("../DatasetBinaryStorage/" + size_string
                                     + "/train/labels" + str(train_pack_number) + ".npz")['arr_0']
    loaded_training_features = np.load("../DatasetBinaryStorage/" + size_string
                                       + "/train/features" + str(train_pack_number) + ".npz")['arr_0']
    return loaded_training_features, loaded_training_labels


def get_current_validation_pack_of_data(validation_pack_number, size_string):
    # If preloaded, just return the preloaded data
    if PRELOAD_DATA and preloaded_val_feat is not None:
        return preloaded_val_feat, preloaded_val_labels

    # Else load it from the disk
    loaded_validation_labels = np.load("../DatasetBinaryStorage/" + size_string
                                       + "/validate/labels" + str(validation_pack_number) + ".npz")['arr_0']
    loaded_validation_features = np.load("../DatasetBinaryStorage/" + size_string
                                         + "/validate/features" + str(validation_pack_number) + ".npz")['arr_0']
    return loaded_validation_features, loaded_validation_labels


PRELOAD_DATA = False
preloaded_val_feat = None
preloaded_val_labels = None
preloaded_train_feat = None
preloaded_train_labels = None


def target_preload_train(train_pack_number, size_string):
    global preloaded_train_feat
    global preloaded_train_labels
    preloaded_train_labels = np.load("../DatasetBinaryStorage/" + size_string
                                     + "/train/labels" + str(train_pack_number) + ".npz")['arr_0']
    preloaded_train_feat = np.load("../DatasetBinaryStorage/" + size_string
                                   + "/train/features" + str(train_pack_number) + ".npz")['arr_0']


def target_preload_validation(validation_pack_number, size_string):
    global preloaded_val_feat
    global preloaded_val_labels
    preloaded_val_labels = np.load("../DatasetBinaryStorage/" + size_string
                                   + "/validate/labels" + str(validation_pack_number) + ".npz")['arr_0']
    preloaded_val_feat = np.load("../DatasetBinaryStorage/" + size_string
                                 + "/validate/features" + str(validation_pack_number) + ".npz")['arr_0']


def preload_training_data(pack_number, size_string):
    global preloaded_train_feat
    global preloaded_train_labels
    if PRELOAD_DATA:
        del preloaded_train_feat, preloaded_train_labels
        t = threading.Thread(target=target_preload_train, args=[pack_number, size_string])
        t.start()


def preload_validation_data(pack_number, size_string):
    global preloaded_val_feat
    global preloaded_val_labels
    if PRELOAD_DATA:
        del preloaded_val_feat, preloaded_val_labels
        t = threading.Thread(target=target_preload_validation, args=[pack_number, size_string])
        t.start()


def train_model(model, model_save_path, batch_size, train_pack_size,
                validation_pack_size, total_epochs, pack_epochs, width, height):
    global preloaded_train_feat
    global preloaded_train_labels
    global preloaded_val_labels
    global preloaded_val_feat

    # Initialize training parameters
    size = f"{width}x{height}"
    train_poz = -1
    validation_poz = -1
    current_epoch = 0
    train_indexes = [x for x in range(train_pack_size)]
    validation_indexes = [x for x in range(validation_pack_size)]
    random.shuffle(train_indexes)
    random.shuffle(validation_indexes)
    last_train_index = train_indexes[0]
    last_validation_index = validation_indexes[0]

    # Data arrays
    loaded_training_features = []
    loaded_training_labels = []
    loaded_validation_features = []
    loaded_validation_labels = []

    # threshold for saving the model
    next_threshold_checkpoint = 1
    threshold = total_epochs // 10 * next_threshold_checkpoint

    # If only one package is available for training
    if train_pack_size == 1:
        # Load Data Packets for Training
        loaded_training_features, loaded_training_labels = get_current_train_pack_of_data(train_indexes[train_poz],
                                                                                          size)
        loaded_validation_features, loaded_validation_labels = get_current_validation_pack_of_data(
            validation_indexes[validation_poz], size)
        # Train & Exit
        return core_train_model(model, model_save_path, loaded_training_features, loaded_training_labels,
                                loaded_validation_features, loaded_validation_labels, total_epochs, batch_size, True)

    # Print current queue for training
    print(train_indexes)
    print(validation_indexes)

    if PRELOAD_DATA:
        train_poz = 0
        validation_poz = 0

    time_x = time_z = time.time()
    while current_epoch < total_epochs:

        # Load & Preload Training Data Packages
        if True:
            del loaded_training_features, loaded_training_labels
            train_poz += 1

            # Shuffle training queue
            if train_poz == train_pack_size:
                train_poz = 0
                random.shuffle(train_indexes)
                print(train_indexes)

            # Load Training Data
            loaded_training_features, loaded_training_labels = get_current_train_pack_of_data(
                train_indexes[train_poz - 1],
                size)

            # Preload Next Training Package
            preload_training_data(train_indexes[train_poz], size)

        # Load & Preload Validation Data Packages
        if validation_pack_size != 1:
            del loaded_validation_labels, loaded_validation_features
            validation_poz += 1

            # Shuffle validation queue
            if validation_poz == validation_pack_size:
                validation_poz = 0
                random.shuffle(validation_indexes)
                print(validation_indexes)

            # Load Validation Data
            loaded_validation_features, loaded_validation_labels = get_current_validation_pack_of_data(
                validation_indexes[validation_poz - 1], size)

            # Preload Next Validation Package
            preload_validation_data(validation_indexes[validation_poz], size)
        elif current_epoch == 0:
            # Load Validation Data
            loaded_validation_features, loaded_validation_labels = get_current_validation_pack_of_data(
                validation_indexes[validation_poz], size)

        # Update Save Checkpoint
        save = False
        if current_epoch > threshold:
            save = True
            next_threshold_checkpoint += 1
            threshold = (total_epochs // 10) * next_threshold_checkpoint

        # Train model
        core_train_model(model, model_save_path, loaded_training_features, loaded_training_labels,
                         loaded_validation_features, loaded_validation_labels, pack_epochs, batch_size, save)

        # Update global epoch progress
        current_epoch += pack_epochs

        # Print Information about the Training Process
        print(f"\nGlobal Epochs: {current_epoch} / {total_epochs}")
        if not PRELOAD_DATA:
            print(f"Trained on pack [{train_indexes[train_poz]}]")
            print(f"Validated on pack [{validation_indexes[validation_poz]}]")
        else:
            print(f"Trained on pack [{last_train_index}]")
            print(f"Validated on pack [{last_validation_index}]")
            last_train_index = train_indexes[train_poz]
            last_validation_index = validation_indexes[validation_poz]
        time_y = time.time()
        print(f"Training Session Time : {time_y - time_x}s")
        print(f"Total Training Time : {time_y - time_z}s")
        time_x = time_y


def get_premade_model(input_shape, outputs_number, lr):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(input_shape[1],
                                                                input_shape[0],
                                                                input_shape[2]), include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = keras.Sequential(
        [
            base_model,
            keras.layers.AveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(outputs_number, activation="softmax")
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])
    return model


def core_train_model(model, model_save_path, train_features, train_labels, validation_features, validation_labels,
                     epochs, batch_size, save):
    # lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss',)
    history = model.fit(x=train_features, y=train_labels, validation_data=(validation_features, validation_labels),
                        epochs=epochs, shuffle=True, batch_size=batch_size)
    if save:
        model.save(model_save_path)
    return history


def plot_model_history(model_history, name):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(model_history['acc'])
    axs[0].plot(model_history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='best')
    # summarize history for loss30
    axs[1].plot(model_history['loss'])
    axs[1].plot(model_history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='best')
    # save
    plt.savefig("../Models&Learning/graphs/" + name + ".png")
    plt.show()


def manage_history(model_history, save, load):
    model_history = {'acc': [float(x) for x in model_history.history['acc']],
                     'val_acc': [float(x) for x in model_history.history['val_acc']],
                     'loss': [float(x) for x in model_history.history['loss']],
                     'val_loss': [float(x) for x in model_history.history['val_loss']]}
    if load:
        with open("history", "r") as f:
            last_history = json.load(f)
            last_history['acc'].extend(model_history['acc'])
            last_history['val_acc'].extend(model_history['val_acc'])
            last_history['loss'].extend(model_history['loss'])
            last_history['val_loss'].extend(model_history['val_loss'])
            model_history = {'acc': last_history['acc'],
                             'val_acc': last_history['val_acc'],
                             'loss': last_history['loss'],
                             'val_loss': last_history['val_loss']}
    if save:
        with open("history", "w+") as f:
            json.dump(
                {'acc': model_history['acc'], 'val_acc': model_history['val_acc'],
                 'loss': model_history['loss'], 'val_loss': model_history['val_loss']},
                f)
    plot_model_history(model_history, MODEL_NAME)


def rescale(x):
    return x / 255.0


def hsv_and_grayscale(x):
    y = tf.image.rgb_to_hsv(x)
    return tf.concat([y, tf.image.rgb_to_grayscale(y)], axis=-1)


def hsv_and_normal(x):
    return tf.concat([x, tf.image.rgb_to_hsv(x)], axis=-1)


def greyscale_and_normal(x):
    return tf.concat([x, tf.image.rgb_to_grayscale(x)], axis=-1)


def grey_norm_hsv(x):
    return tf.concat([x, tf.image.rgb_to_grayscale(x), tf.image.rgb_to_hsv(x)], axis=-1)


def flip_left_right(x):
    return tf.image.random_flip_left_right(x)


def flip_up_down(x):
    return tf.image.random_flip_up_down(x)


def aug_brightness(x):
    return tf.image.random_brightness(x, 0.2)


def rotation(x):
    return  keras.preprocessing.random_rotation(x,20)


def get_model(input_shape, outputs_number, lr):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Lambda(flip_left_right))
    model.add(layers.Lambda(flip_up_down))
    model.add(layers.Lambda(grey_norm_hsv))
    model.add(layers.Lambda(rescale))
    model.add(layers.Conv2D(filters=64, kernel_size=(11, 11), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1024, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1024, activation="relu"))
    model.add(layers.Dense(units=outputs_number, activation="softmax"))
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])
    return model


def load_and_craft_model(input_shape, outputs_number, lr, path):
    return keras.models.load_model(path)


if __name__ == '__main__':
    load_dictionaries()
    # Adam and Adamax
    # HYPER PARAMETERS
    MODEL_ID = 15
    WIDTH = 52
    HEIGHT = 64
    CHANNELS = 3
    BATCH_SIZE = 256
    LABELS = 262
    EPOCHS = 20
    PACK_EPOCHS = 1
    T_PACK_SIZE = 1
    V_PACK_SIZE = 1
    LR = 0.00025
    PRELOAD_DATA = False
    LOAD_MODEL = False
    SAVE_HISTORY = False
    LOAD_HISTORY = False
    MODEL_NAME = f"{WIDTH}x{HEIGHT}_ID{MODEL_ID}"
    size = f"{WIDTH}x{HEIGHT}"

    if LOAD_MODEL:
        model = load_and_craft_model((HEIGHT, WIDTH, CHANNELS), LABELS, LR, "new_gen_models/" + MODEL_NAME)
    else:
        # Build Model
        model = get_model((HEIGHT, WIDTH, CHANNELS), LABELS, LR)
    model.summary()

    # Train Model
    tf.executing_eagerly()
    history = train_model(model=model, model_save_path="new_gen_models/" + MODEL_NAME, batch_size=BATCH_SIZE,
                          width=WIDTH, height=HEIGHT, train_pack_size=T_PACK_SIZE, validation_pack_size=V_PACK_SIZE,
                          total_epochs=EPOCHS, pack_epochs=PACK_EPOCHS)

    manage_history(history, SAVE_HISTORY, LOAD_HISTORY)
