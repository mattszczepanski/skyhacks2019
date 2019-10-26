from fastai.callbacks import SaveModelCallback
from fastai.vision import * # import the vision module
from pathlib import Path

if __name__ == '__main__':
    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05,
                          max_warp=0.)

    src = (ImageList.from_csv(Path('D:\PROGRAMOWANIE\PycharmProjects\skyhacks'),
                              'train.csv', folder='combined', suffix='.jpg')
           # Load data from csv
           .random_split_by_pct(0.2)
           # split data into training and validation set (20% validation)
           .label_from_df(label_delim=' ')
           # label data using the tags column (second column is default)
           )
    data = (src.transform(tfms, size=200)
            # Apply transforms and scale images to 128x128
            .databunch(bs=48).normalize(imagenet_stats)
            # Create databunch with batchsize=64 and normalize the images
            )
    acc_02 = partial(accuracy_thresh, thresh=0.2)
    f_score = partial(fbeta, thresh=0.2)

    learn = create_cnn(data, models.resnet50, metrics=[acc_02, f_score])

    lr = 0.02

    learn.unfreeze()
    learn.fit_one_cycle(5, lr, callbacks=[SaveModelCallback(learn)])
    learn.save('finished')
    learn.load('bestmodel')
    learn.export()