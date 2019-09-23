from typing import Dict
import getopt
import sys
import numpy as np
import cv2
import time
import os

TEST_PICS_DIR = "test_pics"
TRAINING_DIR_CAT = ""

#########################
# Process Params
#########################
opts, args = getopt.getopt(sys.argv[1:], 't:', ['train='])
if len(opts) > 0:
    for opt, arg in opts:
        if opt in ('-t', '--train'):
            TRAINING_DIR_CAT = arg
            print("TRAINING MODE: Outputting examples for " + TRAINING_DIR_CAT)

#########################
# train k nearest neighbors
#########################
train = []
train_labels = []
train_label_dict: Dict[int, str] = {}
cat_list = [ item for item in os.listdir(TEST_PICS_DIR)]

for index, category in enumerate(cat_list):
    # Store category in numbered category dictionary
    train_label_dict[index] = category

    filenames = [file for file in os.listdir(os.path.join(TEST_PICS_DIR, category))]
    # load each file into the data array
    for filename in filenames:
        train.append(cv2.imread(os.path.join(TEST_PICS_DIR, category, filename), 0))
        train_labels.append(index)

# format to be acceptable by knn.train
np_array = np.array(train)
train = np_array.reshape(-1, 320 * 240).astype(np.float32)
train_labels = np.array(train_labels)

# kNN
knn = cv2.ml.KNearest_create() 
print("Training based off of " + str(len(train)) + " images")
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
print("Training complete!")

#########################
# Set up camera capture
#########################
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=50)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


#########################
# Test Current frame against k nearest neighbor model
#########################
def checkCategory(frame):
    print("Checking!")

#########################
# Save to file - returns false if error
#########################
def save_to_file( frame, directory):
    # check if directory exists, and make it if not
    if not os.path.isdir(os.path.join(TEST_PICS_DIR, directory)):
        os.mkdir(os.path.join(TEST_PICS_DIR, directory))
    path = os.path.join(TEST_PICS_DIR, directory, str(time.time()) + ".png")
    print("Saving to: " + path)
    return cv2.imwrite(path, frame)

#########################
# Start camera loop
# 
# 1) apply mask to remove background
# 2) Grascale input
# 3) Set threshold to filter out low bgr values
# 4) Add light from previous frame to current to make a cumulutive gesture picture
#########################

old_frame:list = []

while(True):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(grayscaled, 245, 255, cv2.THRESH_BINARY)

    # Start accumulating frames...
    if len(old_frame) == 0:
        old_frame= np.zeros_like(threshold)
    appended_frame = cv2.addWeighted(threshold,1, old_frame,1,0)

    old_frame = appended_frame
    # cv2.imshow('frame',frame)
    # cv2.imshow('fgmask', fgmask)

    cv2.imshow('threshold', appended_frame)


    key_press = cv2.waitKey(1)
    if key_press == ord('q'):
        break

    elif key_press == ord('b'):
        checkCategory(appended_frame)

    elif key_press == ord('c'):
        old_frame = np.zeros_like(old_frame)
        appended_frame = np.zeros_like(old_frame)

    elif key_press == ord('s'):
        # if not save_to_file(appended_frame, 'circle'):
        if not save_to_file( appended_frame, TRAINING_DIR_CAT):
            raise Exception('Image did not write!')
        else: 
            print("Image saved!")
            old_frame = np.zeros_like(old_frame)
            appended_frame = np.zeros_like(old_frame)

cap.release()
cv2.destroyAllWindows()
