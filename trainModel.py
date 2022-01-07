# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:28:58 2021

@author: priya
"""
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import numpy as np
import pandas as pd
import deeplabcut
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
import ctypes, sys
from datetime import datetime
from pathlib import Path
import glob

def createConfig():
	os.environ["DLClight"]="True"                      
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       
	cwd = os.getcwd()
	print("------------------------")
	video_path = [".....mp4"]
	ProjectFolderName = 'Cam3'
	YourName = 'Priyanka'
	# model2use = 'full_dog'

	path_config_file = deeplabcut.create_pretrained_project(ProjectFolderName, YourName, video_path, videotype="mp4  ", 
	                                            analyzevideo=True, createlabeledvideo=True, copy_videos=True) #must leave copy_videos=True
	return path_config_file

def Train(config_path):
	os.environ["DLClight"]="False"
	test_video_path = ['....avi']
	print("------------1(Train network)--------------")
	# remember, there are several networks you can pick, the default is resnet-50!    
	deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0, max_snapshots_to_keep=100, displayiters=1000, saveiters=10000, maxiters=500000, allow_growth=True)
	print("------------2(Eval network)--------------")
	deeplabcut.evaluate_network(config_path, plotting=True)
	print("------------3(Analyze vids)--------------")
	deeplabcut.analyze_videos(config_path, test_video_path,videotype='avi') 
	print("------------4(Create labeled vids)--------------")
	deeplabcut.create_labeled_video(config_path,test_video_path)
	print("------------5(Plot trajectory)--------------")
	deeplabcut.plot_trajectories(config_path,test_video_path)    
	print("------------6(Get csv data)--------------")
	
	for vid in test_video_path:
		video = vid
		videofolder = str(Path(video).parents[0])
		vname = str(Path(video).stem)
		config_file = auxiliaryfunctions.read_config(path_config_file)
		trainingsetindex=0
		trainFraction = config_file["TrainingFraction"][trainingsetindex]
		DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName( config_file, shuffle = 1, trainFraction=trainFraction, modelprefix="")  

		filtered = False
		track_method=""
		df, _, _, suffix = auxiliaryfunctions.load_analyzed_data(videofolder, vname, DLCscorer, filtered, track_method)
		df_loc = '../'+vname+'-x_y_coords.csv'
		df.to_csv(df_loc)
		df = read_df(df_loc)

		df.loc[df['likelihood'] <= 0.4, 'x'] = np.NAN
		df.loc[df['likelihood'] <= 0.4, 'y'] = np.NAN
		df.loc[df['likelihood.1'] <= 0.4, 'x.1'] = np.NAN
		df.loc[df['likelihood.1'] <= 0.4, 'y.1'] = np.NAN
		df.loc[df['likelihood.2'] <= 0.4, 'x.2'] = np.NAN
		df.loc[df['likelihood.2'] <= 0.4, 'y.2'] = np.NAN
		df.loc[df['likelihood.3'] <= 0.4, 'x.3'] = np.NAN
		df.loc[df['likelihood.3'] <= 0.4, 'y.3'] = np.NAN
		df.loc[df['likelihood.4'] <= 0.4, 'x.4'] = np.NAN
		df.loc[df['likelihood.4'] <= 0.4, 'y.4'] = np.NAN
		df.loc[df['likelihood.5'] <= 0.4, 'x.5'] = np.NAN
		df.loc[df['likelihood.5'] <= 0.4, 'y.5'] = np.NAN


		df = df.interpolate(method = "spline" ,  order=5, limit_direction = 'forward')
		df_loc = '../'+vname+'_smoothed-x_y_coords.csv'
		
		df.to_csv(df_loc,index=False)

def read_df(loc):
	df = pd.read_csv(loc,header=2)
	return df

def add_vids(config_file):
	videos = ['....avi']
	deeplabcut.add_new_videos(config_file, videos, copy_videos=True)

if __name__ == "__main__" :
	path = createConfig()
	# print(path)
	path = "...yaml"
	# adding new videos to train set
	add_vids(path)
	# Move the labelled data to the folder!!!! That's why the below commands are commented for now!
	# also edit the config files according to you're needs before going ahead!!!!!
	deeplabcut.create_training_dataset(path)

	# make sure you update the paths in the Train module :) Copy and paste from the results you have from the prev output
	# PLEASE run the train function as a batch job over GPU. 
	Train(path)
