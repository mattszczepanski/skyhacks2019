import csv
import os
import logging
import random
import numpy as np
from typing import Tuple

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image


from fastai.vision import *  # import the vision module
from pathlib import Path
import pandas as pd

__author__ = 'More Powerful'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_dir = 'data/test_dataset'
answers_file = os.path.join('outputs', 'results.csv')

labels_task_1 = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                 'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                 'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                 'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                 'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                 'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                 'Wall', 'Window']

labels_task2 = ['apartment', 'bathroom', 'bedroom', 'dinning_room', 'house', 'kitchen', 'living_room']

output = []

path = Path('models')
test_il = ImageList.from_folder(Path(input_dir))
learn = load_learner(path, test=test_il)
pfiles = learn.data.test_dl.dataset.items
pfiles = [x.name for x in list(pfiles)]
pf_to_id = dict([(fname, idx) for idx, fname in enumerate(pfiles)])


def load_task_3b_model():
    NUM_CLASSES = 2

    RESNET50_POOLING_AVERAGE = 'avg'
    DENSE_LAYER_ACTIVATION = 'softmax'
    resnet_weights_path = 'models/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    model = Sequential()
    model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights=resnet_weights_path))
    model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
    model.load_weights("models/trained_models/best.hdf5")
    return model



def task_1(partial_output: dict, file_path: str) -> dict:
    logger.debug("Performing task 1 for file {0}".format(file_path))

    filename = file_path.split(os.sep)[-1:][0]
    img_id = pf_to_id[filename]
    found_labels = learn.predict(test_il[img_id])[0].obj
    for label in labels_task_1:
        partial_output[label] = 1 if label in found_labels else 0

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return partial_output


def task_2(file_path: str) -> str:
    logger.debug("Performing task 2 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return labels_task2[random.randrange(len(labels_task2))]


def task_3(model, file_path: str) -> Tuple[str, str]:
    logger.debug("Performing task 3 for file {0}".format(file_path))
    image_size = 224

    img = image.load_img(file_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict_classes(x)[0]
    preds = preds + 3

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return preds, '3'


def main():
    logger.debug("Sample answers file generator")

    task_3b_model = load_task_3b_model()

    for dirpath, dnames, fnames in os.walk(input_dir):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                task_3_output = task_3(task_3b_model, file_path)
                output_per_file = {
                    'filename': f,
                    'task2_class': task_2(file_path),
                    'tech_cond': task_3_output[0],
                    'standard': task_3_output[1]
                }
                output_per_file = task_1(output_per_file, file_path)

                output.append(output_per_file)

    with open(answers_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    main()
