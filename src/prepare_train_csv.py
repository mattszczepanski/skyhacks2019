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
    data_path = Path('../data')
    df = pd.read_csv(data_path / 'labels.csv')
    col_labs = list(df.columns.values)[4:]

    df['tags'] = df[col_labs].apply(lambda x: ts(col_labs, x), axis=1)
    df['Image_name'] = df['filename'].apply(lambda x: x[:-4])
    df = df[['Image_name', 'tags']]
    df.dropna(inplace=True)
    df.to_csv(data_path / 'train.csv', index=False)


get_df()
