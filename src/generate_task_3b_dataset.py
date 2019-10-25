import numpy as np
import pandas as pd

df = pd.read_csv('data/labels.csv')

task_3b_dataset = df[
    np.logical_and(
        np.logical_or(
            df['tech_cond'] == 3,
            df['tech_cond'] == 4),
        df['task2_class'] != 'validation')
]
task_3b_dataset = task_3b_dataset[['filename', 'tech_cond']]
task_3b_dataset['tech_cond'] = task_3b_dataset['tech_cond'] - 3

task_3b_dataset.to_csv('../data/task_3b_labels.csv', index=False)
