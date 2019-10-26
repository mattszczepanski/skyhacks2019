from fastai.vision import * # import the vision module
from pathlib import Path
import pandas as pd

path = Path('D:\PROGRAMOWANIE\PycharmProjects\skyhacks')


if __name__ == '__main__':
    test_il = ImageList.from_folder(Path(r'D:\DANE HAHATON\test_dataset'))
    learn = load_learner(path, test=test_il)
    print("loaded")
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    print("preducting")
    thresh = 0.2
    labelled_preds = [' '.join(
        [learn.data.classes[i] for i, p in enumerate(pred) if p > thresh]) for
                      pred in preds]
    fnames = [f.name[:-4] for f in learn.data.test_ds.items]
    df = pd.DataFrame({'image_name': fnames, 'tags': labelled_preds},
                      columns=['image_name', 'tags'])
    x = 1
