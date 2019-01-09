Notes:

Data is held in tree, raw/cropped train/validation label(left/right) ie data_raw/train/left/img.png

formatting_data.py will iterate through the data_raw, crop faces from the images using cascade and OpenCV and save into the respective folder in data_cropped.

DNN_Face_Orientation.ipynb contains, trains and tests the DNN using the formatted data.

mouse_control.py uses NirCMD software to control mouse from the command line.

live.py accesses the webcam and detects faces real time. The final step for completion is to import functions from mouse_control.py and DNN_face_orientation.ipynb to estimate face orientation, use some form of integral control and then control mouse velocity/position.

Acknowledgements:

Face recognition adapted from article written by Shantnu Tiwari https://realpython.com/blog/python/face-recognition-with-python/
DNN adapted from PyTorch Tutorial - Deep Learning with PyTorch: A 60 Minute Blitz Author: Soumith Chintala



Oliver Rose
@oliverrose1998
