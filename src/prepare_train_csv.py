import glob
from pathlib import Path

import pandas as pd


def ts(col_labs, x):
    res = []
    for i, val in enumerate(list(x)):
        if val:
            res.append(col_labs[i])
    if len(res):
        return ';'.join(res)


def get_df():
    data_path = Path('Downloads')/'SKYHACKS'/'skyhacks_hackathon_dataset'
    images = glob.glob(str(data_path/'training_images/*.jpg'))
    image_names = [x.replace(str(data_path/'training_images/'), '')[1:] for x in images]
    df = pd.read_csv(data_path / 'training_labels.csv')
    df = df.loc[df.Name.isin(image_names)]
    col_labs = list(df.columns.values)[1:]
    df.loc[(df['Building'] == False) & (df['Castle'] == True),
           'Building'] = True
    df.loc[(df['Building'] == False) & (df['Church'] == True), 'Church'] = True
    df.loc[(df['Trees'] == False) & (df['Forest'] == True), 'Trees'] = True
    df['tags'] = df[col_labs].apply(lambda x: ts(col_labs, x), axis=1)
    df['Image_name'] = df['Name'].apply(lambda x: x[:-4])
    df = df[['Image_name', 'tags']]
    df.dropna(inplace=True)
    df.to_csv(data_path / 'train.csv', index=False)


get_df()
