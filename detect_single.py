import numpy as np
import pickle
import cv2
import sys
from color import convert_color
from features import extract_features
from detect_with_labels import cars_from_bboxes, draw_boxes

dist_pickle = pickle.load( open("svc_classifier.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
p = dist_pickle["parameters"]


features = extract_features(sys.argv[1:], color_space=p['color_space'],
                        spatial_size=p['spatial_size'], hist_bins=p['hist_bins'],
                        orient=p['orient'], pix_per_cell=p['pix_per_cell'],
                        cell_per_block=p['cell_per_block'],
                        hog_channel=p['hog_channel'], spatial_feat=p['spatial_feat'],
                        hist_feat=p['hist_feat'], hog_feat=p['hog_feat'])

scaled_features = X_scaler.transform(features)
for prediction in svc.predict(scaled_features):
    print('car' if prediction else 'not car')
