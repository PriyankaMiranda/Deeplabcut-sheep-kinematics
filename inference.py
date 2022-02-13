import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = './temp.csv'
img_folder_loc = '.'



df = pd.read_csv(csv_path, header=None)
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[3:].reset_index(drop=True)
df = df[0:]

new_csv = df

for index, row in new_csv.iterrows():
    # print(index,row)

    x_coord = [float(row[1]), float(row[4]), float(row[7]), float(row[10]), float(row[13]), float(row[16])]
    y_coord = [float(row[2]), float(row[5]), float(row[8]), float(row[11]), float(row[14]), float(row[17])]
    likelihood = [float(row[3]), float(row[6]), float(row[9]), float(row[12]), float(row[15]), float(row[18])]

    for i, likelihood in enumerate(likelihood):
        print(i, likelihood)
        if likelihood < min_liklihood_value:
            row[i*3+1] = np.NaN
            row[i*3+2] = np.NaN
new_csv.to_csv('./new_temp.csv')
