import pandas as pd
import numpy as np
import os


def fusion(fusion1, fusion2, rate1, rate2):
    df_fusion1 = pd.read_csv(fusion1)
    df_fusion2 = pd.read_csv(fusion2)

    for index, row in df_fusion1.iterrows():
        fore1 = row['forecastVolum']
        fore2 = df_fusion2.loc[index, 'forecastVolum']
        df_fusion2.loc[index, 'forecastVolum'] = int(fore1 * rate1 + fore2 * rate2)

    df_fusion2.to_csv('submit.csv', index=False)


if __name__ == '__main__':
    fusion('0.62_fusion_qs.csv', '0.62058.csv', 0.5, 0.5)