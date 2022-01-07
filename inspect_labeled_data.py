import cv2
import numpy as np
import pandas 
import glob, os
import matplotlib.pyplot as plt
folder_loc = ''

os.chdir(folder_loc)
for file in glob.glob("*.csv"):
    file_loc = folder_loc+'/'+file
data = pandas.read_csv(file_loc,header=None)
data = data[1:] #take the data less the header row

data.columns = (data.iloc[0] + '_' + data.iloc[1])
data = data.iloc[2:].reset_index(drop=True)
# print(data.head())

for index, row in data.iterrows():	
	print(row[1])
	print(row[2])
	print(row[3])
	print(row[4])
	print(row[5])
	if(!row[5] == NaN):
		# do processing
	
