import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = './mid/Copy_CollectedData_ISI_Cam-2.csv'
img_folder_loc = '.'

df = pd.read_csv(csv_path, header=None)
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[3:].reset_index(drop=True)
df = df[0:] #RANGE OF IMAGES TO PLOT

for index, row in df.iterrows():
    print(index,row)
    image = row[0]
    image = image.replace('\\', '/')
    image = img_folder_loc+'/'+image

    x_coord = [float(row[1]), float(row[3]), float(row[5]), float(row[7]), float(row[9]), float(row[11])]
    y_coord = [float(row[2]), float(row[4]), float(row[6]), float(row[8]), float(row[10]), float(row[12])]

    print(x_coord, y_coord)

    fig = plt.figure()
    fig.suptitle(f'{row[0]}')
    img = plt.imread(image)
    plt.scatter(x_coord, y_coord)
    plt.imshow(img)

    plt.savefig(fname = f'./plot_{index}.png')
