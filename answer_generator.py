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

class_conf_dict = {
    'Refrigerator': 0.06030190759899247,
    'Tablecloth': 0.07619271434173108,
    'Grass': 0.0807849153127913,
    'Nightstand': 0.08298336884261699,
    'Chandelier': 0.08720960113791561,
    'Shower': 0.09123444721531762,
    'Chest of drawers': 0.09317955689643795,
    'Drawer': 0.0950834843104803,
    'Rural area': 0.09877726671883286,
    'Fireplace': 0.10057131145144356,
    'Urban area': 0.10233260797103441,
    'Cupboard': 0.10743596585062351,
    'Sky': 0.12570050205590746,
    'Bathtub': 0.12984281590613014,
    'Bathroom sink': 0.1311924964259911,
    'Kitchen & dining room table': 0.13899555562930258,
    'Kitchen stove': 0.14515285869102879,
    'Window': 0.14871418739556916,
    'Door': 0.15332185385017896,
    'Tree': 0.15778153063947187,
    'Bathroom cabinet': 0.1588747959804537,
    'Tap': 0.1642198320526484,
    'Facade': 0.16526567952580895,
    'Coffee table': 0.1683595016657622,
    'Toilet': 0.17238744029201253,
    'Chair': 0.18479354350932545,
    'Curtain': 0.18479354350932856,
    'Roof': 0.18752293919156546,
    'Mattress': 0.20218319120289496,
    'Couch': 0.21719184712566733,
    'Sink': 0.22460287433453033,
    'Hardwood': 0.23102038866945515,
    'Plumbing fixture': 0.2869175941071937,
    'Dining room': 0.30903645814569963,
    'Tile': 0.32462785611570505,
    'Room': 0.32633048218545124,
    'Countertop': 0.32675373374126687,
    'Cabinetry': 0.3309343517141135,
    'Bed frame': 0.33298998517581796,
    'Bed sheet': 0.33902321535115104,
    'Kitchen': 0.35015049345375365,
    'Property': 0.35636588962600474,
    'Bathroom': 0.35672476944516546,
    'House': 0.3847491563804744,
    'Table': 0.3972542820622387,
    'Bed': 0.39753304997747707,
    'Wall': 0.41603210209131347,
    'Ceiling': 0.4160321020913136,
    'Real estate': 0.4165179695452889,
    'Bedroom': 0.41891978456561496,
    'Living room': 0.42220639657003967,
    'Furniture': 0.44383797361977606,
    'Floor': 0.49118487285743256
}


def generate_task_1_predictions():
    path = Path('models')
    test_il = ImageList.from_folder(Path(input_dir))
    learn = load_learner(path, test=test_il)
    predictions, _ = learn.get_preds(ds_type=DatasetType.Test)

    labeled_preds = {}

    for path, pred in zip(test_il.items, predictions):
        current_class_output = []
        for i, p in enumerate(pred):
            curr_class = learn.data.classes[i]
            if p > class_conf_dict.setdefault(curr_class, 0)*0.1:
                current_class_output.append(curr_class)
        labeled_preds[str(path)] = current_class_output
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

    print(task_1_predictions)

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
