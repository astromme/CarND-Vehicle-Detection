import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import pickle
from lesson_functions import get_hog_features, extract_features

from cars_notcars import cars, notcars

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
# sample_size = 50
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
p = {
    'color_space': 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 9,  # HOG orientations
    'pix_per_cell': 8, # HOG pixels per cell
    'cell_per_block': 2, # HOG cells per block
    'hog_channel': "ALL", # Can be 0, 1, 2, or "ALL"
    'spatial_size': (16, 16), # Spatial binning dimensions
    'hist_bins': 16,    # Number of histogram bins
    'spatial_feat': True, # Spatial features on or off
    'hist_feat': True, # Histogram features on or off
    'hog_feat': True, # HOG features on or off
}

t=time.time()
print('extracting car features')
car_features = extract_features(cars, color_space=p['color_space'],
                        spatial_size=p['spatial_size'], hist_bins=p['hist_bins'],
                        orient=p['orient'], pix_per_cell=p['pix_per_cell'],
                        cell_per_block=p['cell_per_block'],
                        hog_channel=p['hog_channel'], spatial_feat=p['spatial_feat'],
                        hist_feat=p['hist_feat'], hog_feat=p['hog_feat'])
print('extracting notcar features')
notcar_features = extract_features(notcars, color_space=p['color_space'],
                        spatial_size=p['spatial_size'], hist_bins=p['hist_bins'],
                        orient=p['orient'], pix_per_cell=p['pix_per_cell'],
                        cell_per_block=p['cell_per_block'],
                        hog_channel=p['hog_channel'], spatial_feat=p['spatial_feat'],
                        hist_feat=p['hist_feat'], hog_feat=p['hog_feat'])
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',p['orient'],'orientations',p['pix_per_cell'],
    'pixels per cell and', p['cell_per_block'],'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

with open('svc_classifier.p', 'wb') as f:
    pickle.dump({
        'svc': svc,
        'parameters': p,
        'X_scaler': X_scaler,

    }, f)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
