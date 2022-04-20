import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import transform
import cv2
import imageio
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa

# Read and store config file params
with open("preprocess_config.yaml", "r") as yamlfile:
    try:
        data = yaml.load(yamlfile, Loader=yaml.SafeLoader)
        print("Read successful")
    except:
        print("Failed to load config")
        pass

directory = data['folder_location']
preprocessing_params = {}
preprocessing_params['is_zoom'] = (data['preprocessing']['zoom']['is_on'])
preprocessing_params['zoom_factor'] = (data['preprocessing']['zoom']['zoom_factor'])
preprocessing_params['is_brightness'] = (data['preprocessing']['brightness']['is_on'])
preprocessing_params['brightness_factor'] = (data['preprocessing']['brightness']['brightness_factor'])
preprocessing_params['is_translation'] = (data['preprocessing']['translation']['is_on'])
preprocessing_params['translation_coord'] = (data['preprocessing']['translation']['translation_coord'])
preprocessing_params['is_rotation'] = (data['preprocessing']['rotation']['is_on'])
preprocessing_params['rotation_degree'] = (data['preprocessing']['rotation']['rotation_degree'])
preprocessing_params['is_shearing'] = (data['preprocessing']['shearing']['is_on'])
preprocessing_params['shear_degree'] = (data['preprocessing']['shearing']['shear_degree'])

#Read CSV and make DF (Make sure there is only one CSV file in data folder)
csv_path = list(filter(lambda f: f.endswith('.csv'), os.listdir(directory))) #finds all csv files
csv_path = os.path.join(directory, csv_path[0]) #gets the file path to the first csv file
img_folder_loc = directory

#DF cleanup
df = pd.read_csv(csv_path, header=None)
df.columns = (df.iloc[0] + '_' + df.iloc[1])
df = df.iloc[1:].reset_index(drop=True)
df = df[3:]

#Create subfolders for saving preprocessed images
preprocess_directory = os.path.join(directory, 'preprocessing')
if not os.path.exists(preprocess_directory):
    os.makedirs(preprocess_directory)
save_directory = os.path.join(preprocess_directory, str('UNSPECIFIED'))
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

#Save config params in text file
with open(save_directory+"/preprocessing_params.txt", "w") as datafile:
    print(preprocessing_params, file=datafile)

#Extract images and transform
for index, row in df.iterrows():
    image_path = row[0]
    image_path = image_path.replace('\\', '/') #Modify image path to work with python
    image_name = image_path[23:] #Clean image path, might need to be modified with new folder structure formatting
    image_path = img_folder_loc+'/'+image_name
    #Get keypoint coordinates (modify per csv file)
    x_coord = [float(row[1]), float(row[3]), float(row[5]), float(row[7]), float(row[9]), float(row[11])]
    y_coord = [float(row[2]), float(row[4]), float(row[6]), float(row[8]), float(row[10]), float(row[12])]

    # Zips and removes all keypoints with "NaN"
    column_names = ["rightstifle", "righthock", "righthoof", "leftstifle", "lefthock", "lefthoof"]
    coords = {}
    print(coords)

    for idx, x in enumerate(x_coord):
        coords[column_names[idx]] = ([x, y_coord[idx]])
    # Open Image
    img = cv2.imread(image_path)
    # print("Transforming " + image_path)

    #Transform image
    transformed_img, transformed_coords = transform(img, coords, preprocessing_params)
    print(coords, transformed_coords)
    #Save image and add new keypoints to new CSV
    save_path = os.path.join(save_directory, f'{image_name}')
    cv2.imwrite(save_path, transformed_img)
    print("Saving Image", save_path)



    break

print("All Done!")
