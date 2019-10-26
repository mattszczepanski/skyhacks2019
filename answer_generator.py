import csv
import os
import logging
import random
import numpy as np
import pandas as pd
from typing import Tuple

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

from fastai.vision import *  # import the vision module
from pathlib import Path

__author__ = 'More Powerful'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_dir = 'data/test_dataset'
answers_file = os.path.join('outputs', 'results.csv')

LABELS_TASK_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                 'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                 'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                 'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                 'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                 'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                 'Wall', 'Window']

labels_task2 = ['apartment', 'bathroom', 'bedroom', 'dinning_room', 'house', 'kitchen', 'living_room']

output = []


def generate_task_1_predictions():
    path = Path('models')
    test_il = ImageList.from_folder(Path(input_dir))
    learn = load_learner(path, test=test_il)
    predictions, _ = learn.get_preds(ds_type=DatasetType.Test)
    threshold = 0.3

    labeled_preds = {
        str(path): [learn.data.classes[i] for i, p in enumerate(pred) if p > threshold]
        for path, pred in zip(test_il.items, predictions)
    }

    return labeled_preds


def load_task_3b_model():
    NUM_CLASSES = 2

    RESNET50_POOLING_AVERAGE = 'avg'
    DENSE_LAYER_ACTIVATION = 'softmax'
    resnet_weights_path = 'models/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    model = Sequential()
    model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights=resnet_weights_path))
    model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
    model.load_weights("models/trained_models/the_best_model.hdf5")
    return model


def task_1(all_predictions, file_path: str) -> dict:
    logger.debug("Performing task 1 for file {0}".format(file_path))
    partial_output = {}

    found_labels = all_predictions[file_path]
    for label in LABELS_TASK_1:
        partial_output[label] = 1 if label in found_labels else 0

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return partial_output


def task_2(partial_output, classifier, factor, file_path: str) -> str:
    logger.debug("Performing task 2 for file {0}".format(file_path))
    """
    if "dom" in file_path:
        response = "house"
    elif "jadalnia" in file_path:
        response = "dinning_room"
    elif "kuchnia" in file_path:
        response = "kitchen"
    elif "lazienka" in file_path:
        response = "bathroom"
    elif "salon" in file_path:
        response = "living_room"
    elif "sypialnia" in file_path:
        response = "bedroom"
    """
    if False:
        pass
    else:
        features = pd.DataFrame(partial_output, index=[0])[LABELS_TASK_1]
        reversefactor = dict(zip(range(6), factor[1]))
        y_pred = classifier.predict(features)
        y_pred = np.vectorize(reversefactor.get)(y_pred)
        response = y_pred[0]

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return response


def task_3(model, task2_label, file_path: str) -> Tuple[str, str]:
    logger.debug("Performing task 3 for file {0}".format(file_path))

    if task2_label == 'dinning_room' or task2_label == 'house':
        preds = '3'
    else:
        image_size = 224

        img = image.load_img(file_path, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)[0][1]
        preds = int(preds >= 0.5)
        preds = preds + 3

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return preds, '3'


def main():
    logger.debug("Sample answers file generator")

    task_1_predictions = generate_task_1_predictions()
    task_2_classifier = pickle.load(open('models/randomforestmodel.pkl', 'rb'))
    task_2_factor = pickle.load(open('models/factor.pkl', 'rb'))
    task_3b_model = load_task_3b_model()

    for dirpath, dnames, fnames in os.walk(input_dir):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                task_1_output = task_1(task_1_predictions, file_path)
                task_2_label = task_2(task_1_output, task_2_classifier, task_2_factor, file_path)
                task_3_output = task_3(task_3b_model, task_2_label, file_path)
                output_per_file = {
                    'filename': f,
                    'task2_class': task_2_label,
                    'tech_cond': task_3_output[0],
                    'standard': task_3_output[1]
                }
                output_per_file.update(task_1_output)

                output.append(output_per_file)

    with open(answers_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + LABELS_TASK_1)
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    main()
