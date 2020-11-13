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

def generate_df(image_names, column_names, values=1):
    res = []
    for image_name in image_names:
        d= {"Name":image_name}
        for col in column_names:
            d[col]=values
        res.append(d)
    return pd.DataFrame(res)
def get_df():
    data_path = Path('Downloads')/'SKYHACKS'/'skyhacks_hackathon_dataset'

    live_test_images = glob.glob(str(data_path/'live_test_images/*.jpg'))
    live_test_image_names = [x.replace(str(data_path/'live_test_images/'), '')[1:] for x in live_test_images]
    df = pd.read_csv(data_path / 'training_labels.csv')
    df = df.loc[df.Name.isin(live_test_image_names)]
    output_cols = df.columns.values
    col_labs = list(df.columns.values)[1:]
    output_df = generate_df(live_test_image_names, col_labs)
    output_df = output_df.sort_values(by='Name')
    output_df.to_csv(data_path / 'output.csv', index=False)
    z = 1

get_df()
