import os
import sys
import numpy as np

import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dataset

'''Function to crop faces of people within saved image. The function takes the image path as an arguement and returns numpy array corresponding to the cropped image. 

Code adapted from article written by Shantnu Tiwari https://realpython.com/blog/python/face-recognition-with-python/
'''

def crop_face(image):
    faces = []

    # Get user supplied values
    imagePath = image 
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #print("Found {0} faces!".format(len(faces)))

    # Produce 3x32x32 images of the faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:    
            # Crop faces within image
            cropped = image[y:(y+h), x:(x+w)]
    
            # Resize image to 3x32x32
            dim = (32,32)
            resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
        
            #Return cropped image        
            return resized
       
    #If no faces are detected return False
    else:
        return False
       
''' Function formats the raw data (images captured from video using VLC media player) organised in folders for train/validation sets and under appropriate labels ie left/right. The function places cropped images into respective folders in folder for cropped data.  '''

def format_data(raw_root = 'data_raw', cropped_root = 'data_cropped'):   
    
    list_directory = os.listdir('./'+raw_root)
    
    #list_directory includes train and test sets.
    for directory in list_directory:
        list_subdirectory = os.listdir('./'+raw_root+'/'+directory)
        
        #list_subdirectory includes folders of images sorted by label ie left, right etc.
        for subdirectory in list_subdirectory:
            for subdir, dirs, files in os.walk('./'+raw_root+'/'+directory+'/'+subdirectory):
                n=m=0
                
                #files are png images themselves. Images are cropped and if a face is found (the crop_face function will return a numpy array rather than a boolean) then the image is saved in respective folder in data_cropped.
                for file in files:
                    n += 1
                    face = crop_face(raw_root+'/'+directory+'/'+subdirectory+'/'+file)
                    if type(face) != bool:
                        cv2.imwrite(cropped_root+'/'+directory+'/'+subdirectory+'/'+file,face)
                        m += 1
                
                #Print at completion of each folder the number of faces extracted from the folder of png images.
                print('Finished: ',cropped_root+'/'+directory+'/'+subdirectory,'    ',m,'/',n,'Images cropped')
                 
    print('Completed')
    
