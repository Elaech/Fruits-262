import json
import os.path
import sys
import random
import threading
import time
import math
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from PIL import ImageTk
import PIL.Image as PIL_Image
import tensorflow.keras.layers as layers
import subprocess
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog

label_to_number_dict = None
number_to_label_dict = None
model_loaded = False
model = None
image_loaded = False
image = None
image_fig = None
predict_fig = None
width = 52
height = 64
multiplier = 100
top_x_results = 5

default_model = os.path.abspath("../Models&Learning/new_gen_models/52x64_ID19")
base_dataset_path = os.path.abspath("../HandFilteredDataset")


def load_dictionaries():
    global label_to_number_dict
    global number_to_label_dict
    with open("../Organizing/labels/label_to_number_dict.json", "r") as f:
        label_to_number_dict = json.load(f)
    with open("../Organizing/labels/number_to_label_dict.json", "r") as f:
        number_to_label_dict = json.load(f)


def resize_one_image(img, target_dim):
    # If dimensions are different from the target
    if img.size[0] != target_dim[0] or img.size[1] != target_dim[1]:

        # Calculate ratios and differences between image dims
        dim_to_max = 1 * (img.size[1] / 16 > img.size[0] / 13)
        ratios = [img.size[1] / img.size[0], img.size[0] / img.size[1]]

        # Create an image copy for resizing
        img_copy = img.copy()

        # The image is definitely smaller than the target
        if img.size[0] < target_dim[0] and img.size[1] < target_dim[1]:
            new_sizes = [0, 0]
            new_sizes[dim_to_max] = target_dim[dim_to_max]
            new_sizes[(dim_to_max + 1) % 2] = math.floor(ratios[dim_to_max] * target_dim[dim_to_max])
            img_copy = img_copy.resize(tuple(new_sizes), PIL_Image.ANTIALIAS)
        # The image has at least one dimension bigger than the target
        else:
            img_copy.thumbnail(tuple(target_dim), PIL_Image.ANTIALIAS)

        # Create new blank image for resized image insertion
        img = PIL_Image.new("RGB", (target_dim[0], target_dim[1]), (255, 255, 255))
        img.paste(img_copy, ((target_dim[0] - img_copy.size[0]) // 2, (target_dim[1] - img_copy.size[1]) // 2))
    return img


def one_image_prediction_results(result):
    global predict_fig
    predict_fig = plt.figure("Fruit Predictions")
    first = result[0]
    path_to_fruit = f'\"{base_dataset_path}\\{first[0]}\"'
    subprocess.Popen(f'explorer {path_to_fruit}')
    time.sleep(1)
    top = result[:top_x_results]
    labels = tuple([x[0] for x in top])
    values = [100 * x[1] for x in top]
    y_pos = np.arange(len(labels))  # the label locations
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Credibility')
    plt.title('Prediction Results')
    plt.show()


def predict_one_image():
    # Image
    image_array = np.repeat(np.array([np.array(image)]), [multiplier], axis=0)

    # Predict
    raw_results = model.predict(image_array)

    single_result = np.sum(raw_results, axis=0) / multiplier

    # Map to Label and Accuracy Pair
    results_and_labels = [[number_to_label_dict[str(index)], x] for index, x in enumerate(single_result)]

    # Sort by Accuracy
    sorted_results = sorted(results_and_labels, key=lambda x: x[1], reverse=True)

    return sorted_results


def predict_on_click(error_label, top, multi):
    global model_loaded
    global image_loaded
    global multiplier
    global top_x_results
    global predict_fig
    if model_loaded and image_loaded:
        try:
            error_label.config(text="Predicting...")
            multiplier = int(multi)
            top_x_results = int(top)
            if multiplier <= 0 or top_x_results <= 0:
                raise ArithmeticError
            if top_x_results >= 262:
                raise ArithmeticError

            # Predict Image
            results = predict_one_image()
            error_label.config(text="Completed Prediction!")

            # Close older figs
            if predict_fig is not None:
                plt.close(predict_fig)

            # Statistics and Graphs
            one_image_prediction_results(results)

        except:
            error_label.config(text="Error! Recheck Parameters!")
    else:
        error_label.config(text="Parameters not set!")


def make_predict_button(root, top_entry, multi_entry, font, loaded_resource_font, padding):
    predict_label = Label(root, text="", font=loaded_resource_font)
    button = Button(root, text="Predict",
                    command=lambda: predict_on_click(predict_label, top_entry.get(), multi_entry.get()))
    button.config(font=font, cursor="hand2", bg="black", fg="white")
    button.pack(pady=padding)
    predict_label.pack()


def load_image(root, font, image_text_label):
    global image
    global image_loaded
    global image_fig
    root.filename = filedialog.askopenfilename(initialdir=".", title="Select An Image",
                                               filetypes=(("jpg files", "*.jpg"), ("jpeg files", "*.jpeg")))
    try:
        image_text_label.config(text=root.filename, font=font)
        image = PIL_Image.open(root.filename)
        image = resize_one_image(image, (width, height))
        if image_fig is not None:
            plt.close(image_fig)
        image_fig = plt.figure("Resized Image Input")
        image_loaded = True
        plt.imshow(image)
        plt.show()
    except:
        image_text_label.config(text="Failed to load image")
        image = None
        image_loaded = False


def make_load_image_button(root, font, loaded_resource_font, padding):
    image_text_label = Label(root, text="No Image Loaded", font=loaded_resource_font)
    button = Button(root, text="Select Image", command=lambda: load_image(root, loaded_resource_font, image_text_label))
    button.config(font=font, cursor="hand2", bg="black", fg="white")
    button.pack(pady=padding)
    image_text_label.pack()


def load_model(root, font, model_text_label):
    global model
    global model_loaded
    root.filename = filedialog.askopenfilename(initialdir=".", title="Select A Model",
                                               filetypes=(("all files", "*.*"),))
    try:
        model_text_label.config(text=root.filename, font=font)
        model = keras.models.load_model(root.filename)
        model_loaded = True
    except:
        model_text_label.config(text="Failed to load model")
        model = None
        model_loaded = False


def load_default_model(font, model_text_label):
    global model
    global model_loaded
    try:
        model_text_label.config(text=default_model, font=font)
        model = keras.models.load_model(default_model)
        model_loaded = True
    except ArithmeticError:
        model_text_label.config(text="Failed to load default model")
        model = None
        model_loaded = False


def make_load_model_button(root, font, loaded_resource_font, padding):
    model_text_label = Label(root, text="", font=loaded_resource_font)
    button = Button(root, text="Select Model", command=lambda: load_model(root, loaded_resource_font, model_text_label))
    button.config(font=font, cursor="hand2", bg="black", fg="white")
    button.pack(pady=padding)
    model_text_label.pack()
    load_default_model(loaded_resource_font, model_text_label)


def set_input_size(size_string):
    global width
    global height
    if size_string == "13x16":
        width = 13
        height = 16
    elif size_string == "26x32":
        width = 26
        height = 32
    elif size_string == "52x64":
        width = 52
        height = 64
    elif size_string == "104x128":
        width = 104
        height = 128
    else:
        width = 208
        height = 256


def make_size_dropdown(root, font, loaded_resource_font, padding):
    text_label = Label(root, text="Model Input Size", font=loaded_resource_font)
    clicked = StringVar()
    clicked.set("52x64")
    options = ["13x16", "26x32", "52x64", "104x128", "208x256"]
    dropdown = OptionMenu(root, clicked, *options, command=set_input_size)
    dropdown.config(bg="black", fg="white", font=font, cursor="hand2")
    dropdown["menu"].config(bg="black", fg="white", font=font, cursor="hand2")
    dropdown.pack(pady=padding)
    text_label.pack()


def make_multiplier_input(root, font, loaded_resource_font, padding):
    text_label = Label(root, text="Number of Predictions per Image", font=loaded_resource_font)
    entry = Entry(root)
    entry.config(bg="black", fg="white", insertbackground="white", justify=CENTER, font=font)
    entry.insert(END, str(multiplier))
    entry.pack(pady=padding)
    text_label.pack()
    return entry


def make_top_x_input(root, font, loaded_resource_font, padding):
    text_label = Label(root, text="Show top x of Predictions per Image", font=loaded_resource_font)
    entry = Entry(root)
    entry.config(bg="black", fg="white", insertbackground="white", justify=CENTER, font=font)
    entry.insert(END, str(top_x_results))
    entry.pack(pady=padding)
    text_label.pack()
    return entry


def load_gui():
    main_font = ("Arial", 16)
    main_font_bold = ("Arial bold", 16)
    loaded_resource_font = ("Arial bold", 12)
    padding = (25, 5)
    root = Tk()
    root.resizable(False, False)
    root.title("Classify Fruit Images")
    root.iconbitmap("other resources/pineapple.ico")
    root.geometry("800x600")
    make_load_image_button(root, main_font_bold, loaded_resource_font, padding)
    make_load_model_button(root, main_font_bold, loaded_resource_font, padding)
    make_size_dropdown(root, main_font, loaded_resource_font, padding)
    multi_entry = make_multiplier_input(root, main_font, loaded_resource_font, padding)
    top_entry = make_top_x_input(root, main_font, loaded_resource_font, padding)
    make_predict_button(root, top_entry, multi_entry, main_font_bold, loaded_resource_font, padding)
    root.mainloop()


if __name__ == '__main__':
    load_dictionaries()
    load_gui()
