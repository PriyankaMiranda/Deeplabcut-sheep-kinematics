import yaml 
import cv2
import os
from os import walk
import pandas as pd
from pathlib import Path
import io

def readConfig(config):
	return yaml.safe_load(open(config))

def loadConfig(config):
	with open(config, 'r') as fin:
		config_file = yaml.load(fin)
	return config_file

def writeConfig(config, yaml_file):

	if yaml_file:
		with io.open(config, 'w', encoding='utf8') as outfile:
			yaml.safe_dump(dict(yaml_file), outfile, default_flow_style=False, allow_unicode=True)
    
def checkFilePath(path):
	if(os.path.isfile(path)):
		return True
	else:
		return False
def Message(type):
	types= {1: "Config file not found. Check location in my_config. Creating new project for now.",
	2: "Extracting frames.",
	3: "Labeling frames."}
	print(types[type])

def fixFormat(path):
	return os.path.abspath(path) 

def getAVIFilesInFolder(path):
	files =  readFilesInFolder(path)	
	AVIfiles = []
	for filename in files:
		if filename.endswith('.avi'):
			AVIfiles.append(path+"\\"+filename)
	return AVIfiles

def readFilesInFolder(path):
	return next(walk(path), (None, None, []))[2] 

def splitVidData(files):
	cam1filelist, cam2filelist, cam3filelist = [], [], []
	for filename in files:
		if 'Cal' in filename:
			continue
		if 'Cam1' in filename:
			cam1filelist.append(filename)
		elif 'Cam2' in filename:
			cam2filelist.append(filename)
		elif 'Cam3' in filename:
			cam3filelist.append(filename)
	return cam1filelist, cam2filelist, cam3filelist

def countFrames(videos, override=False):
	for path in videos:
		video = cv2.VideoCapture(path)
		total = 0
	try:
		total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	except:
		print("Error counting frames")
		video.release()
		return 0
	return total


def delete_unwanted_entires():
	header , my_df = read_df()
	first_column = my_df[my_df.columns[0]]
	# for row in first_column:
	for index, row in my_df.iterrows():
		img_loc = '..'+row[0].split('\\')[-1]
		my_file = Path(img_loc)
		if(my_file.exists()):
			print("")
		else:
			my_df = my_df.drop(index=index)
	orig_df = my_df
	my_df.columns = header.columns

	top_row = pd.DataFrame(header.iloc[:2])
	my_df = pd.concat([top_row, my_df]).reset_index(drop = True) 
	my_df.to_csv("../CollectedData_Priyanka.csv", index = False)
	orig_df.to_hdf("../CollectedData_Priyanka.h5", key="df_with_missing", mode="w")

def read_df():
	header = pd.read_csv('../CollectedData_Priyanka.csv').iloc[:2]
	df = pd.read_csv('../CollectedData_Priyanka.csv',header=2)
	return header, df
