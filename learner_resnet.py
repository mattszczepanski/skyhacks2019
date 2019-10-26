from fastai.callbacks import SaveModelCallback
from fastai.vision import * # import the vision module
from pathlib import Path
import pandas as pd

path = Path('D:\PROGRAMOWANIE\PycharmProjects\skyhacks\main')
def ts(col_labs, x):
    res = []
    for i, val in enumerate(list(x)):
        if val:
            res.append(col_labs[i])
    if len(res):
        return ' '.join(res)

def get_df():
    df = pd.read_csv(path / 'labels.csv')
    col_labs = list(df.columns.values)[4:]

    df['tags'] = df[col_labs].apply(lambda x: ts(x), axis=1)
    df['Image_name'] = df['filename'].apply(lambda x: x[:-4])
    df = df[['Image_name', 'tags']]
    df.dropna(inplace=True)
    df.to_csv('train_v2.csv', index=False)

if __name__ == '__main__':
    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05,
                          max_warp=0.)

    np.random.seed(
        42)  # set random seed so we always get the same validation set
    src = (ImageList.from_csv(Path('D:\PROGRAMOWANIE\PycharmProjects\skyhacks'),
                              'train_v2.csv', folder='combined', suffix='.jpg')
           # Load data from csv
           .random_split_by_pct(0.2)
           # split data into training and validation set (20% validation)
           .label_from_df(label_delim=' ')
           # label data using the tags column (second column is default)
           )
    data = (src.transform(tfms, size=200)
            # Apply transforms and scale images to 128x128
            .databunch(bs=32).normalize(imagenet_stats)
            # Create databunch with batchsize=64 and normalize the images
            )
    print("started gowno")
    acc_02 = partial(accuracy_thresh, thresh=0.2)
    f_score = partial(fbeta, thresh=0.2)
    # create cnn with the resnet50 architecture
    learn = create_cnn(data, models.resnet50, metrics=[acc_02, f_score])
    # lr = 0.01  # chosen learning rate
    # learn.fit_one_cycle(4, lr)  # train model for 4 epochs
    #
    # learn.save('stage-2')  # save model
    lr = 2e-1
    # #
    # #
    # #
    learn.unfreeze()
    learn.fit_one_cycle(5, lr, callbacks=[SaveModelCallback(learn)])
    learn.save('finished')
    learn.export()