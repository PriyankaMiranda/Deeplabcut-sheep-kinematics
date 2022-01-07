# PREPROCESS IMAGES BASED ON PARAMETERS PROVIDED
import yaml
from PIL import Image, ImageEnhance
import cv2
import imutils
import numpy as np
import os

# READ CONFIG FILE
with open("preprocess_config.yaml", "r") as yamlfile:
    try:
        data = yaml.load(yamlfile, Loader=yaml.SafeLoader)
        print("Read successful")
    except:
        print("Failed to load config")
        pass

directory = data['folder_location']

#create subfolder for preprocessed frames
preprocess_directory = os.path.join(directory, 'preprocessing')
if not os.path.exists(preprocess_directory):
    os.makedirs(preprocess_directory)

is_zoom = (data['preprocessing']['zoom']['is_on'])
zoom_factor = (data['preprocessing']['zoom']['zoom_factor'])

is_brightness = (data['preprocessing']['brightness']['is_on'])
brightness_factor = (data['preprocessing']['brightness']['brightness_factor'])

is_translation = (data['preprocessing']['translation']['is_on'])
translation_coord = (data['preprocessing']['translation']['translation_coord'])

is_rotation = (data['preprocessing']['rotation']['is_on'])
rotation_degree = (data['preprocessing']['rotation']['rotation_degree'])

is_shearing = (data['preprocessing']['shearing']['is_on'])
shear_pixel_displacement = (data['preprocessing']['shearing']['shear_pixel_displacement'])

#initialize dictionary to store transformation values
dict_transforms = {}


# FUNCTIONS
# SCALING/ZOOM
def zoom(img, zoom_factor):
    #calculate dimensions for cropping
    new_height = img_height//zoom_factor
    new_width = img_width//zoom_factor

    #calculate pixel coordinates on original image to make cropped image centered
    new_height_from = img_height//2 - new_height//2
    new_height_to = img_height//2 + new_height//2
    new_width_from = img_width//2 - new_width//2
    new_width_to = img_width//2 + new_width//2
    cropped_img = img[int(new_height_from):int(new_height_to),int(new_width_from):int(new_width_to)]

    #return cropped image to dimensions of original image
    zoomed_img = cv2.resize(cropped_img, (img_width, img_height), interpolation = cv2.INTER_NEAREST)
    return zoomed_img

def brightness(img, brightness_factor):
    #image brightness enhancer
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_factor)

def translation(img, translation_coord):
    M = np.float32([
	   [1, 0, translation_coord[0]],
	   [0, 1, translation_coord[1]]
    ])
    translated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated_img

def rotation(img, rotation_degree):
    rotated_img = imutils.rotate(img,rotation_degree)
    return rotated_img

def shearing(img, shear_pixel_displacement):
    rows,cols,ch = img.shape
    #Shearing along x-axis
    pts1 = np.float32([[0,0],[img_width,0],[0, img_height]])
    pts2 = np.float32([[0+shear_pixel_displacement,0],[img_width+shear_pixel_displacement,0],[0,img_height]])

    M = cv2.getAffineTransform(pts1,pts2)

    sheared_img = cv2.warpAffine(img,M,(cols,rows))
    return sheared_img

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # READ IMAGES
        file_path = os.path.join(directory, filename)
        img = cv2.imread(file_path)
        img_height = img.shape[0]
        img_width = img.shape[1]

        print("Transforming " + filename)

        if is_zoom==True:
            img = zoom(img, zoom_factor)
            dict_transforms['Z'] = zoom_factor
        if is_brightness==True:
            img = brightness(img, brightness_factor)
            img = np.array(img)
            dict_transforms['B'] = brightness_factor
        if is_translation == True:
            img = translation(img, translation_coord)
            dict_transforms['T'] = translation_coord
        if is_rotation == True:
            img = rotation(img, rotation_degree)
            dict_transforms['R'] = rotation_degree
        if is_shearing==True:
            img = shearing(img, shear_pixel_displacement)
            dict_transforms['S'] = shear_pixel_displacement

        save_directory = os.path.join(preprocess_directory, str(dict_transforms))
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        save_path = os.path.join(save_directory, f'{filename}')
        cv2.imwrite(save_path, img)

    else:
        continue

print("Finished Preprocessing")

