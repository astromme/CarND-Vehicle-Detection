import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
from detect_with_labels import cars_from_bboxes, draw_bboxes

dist_pickle = pickle.load( open("svc_classifier.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
p = dist_pickle["parameters"]

image = mpimg.imread('test_images/test1.jpg')

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat,
              hog_feat, hog_channel):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space='YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step


    if hog_channel == 'ALL':
        img_hog_features = []
        for channel in range(ctrans_tosearch.shape[2]):
            img_hog_features.append(get_hog_features(ctrans_tosearch[:,:,channel],
                                orient, pix_per_cell, cell_per_block, feature_vec=False))
    else:
        img_hog_features = get_hog_features(ctrans_tosearch[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            #1) Define an empty list to receive features
            img_features = []

            #3) Compute spatial features if flag is set
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                #4) Append features to list
                img_features.append(spatial_features)
            #5) Compute histogram features if flag is set
            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                #6) Append features to list
                img_features.append(hist_features)
            #7) Compute HOG features if flag is set
            if hog_feat == True:
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(ctrans_tosearch.shape[2]):
                        hog_features.extend(img_hog_features[channel][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())

                else:
                    hog_features = img_hog_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                #8) Append features to list
                img_features.append(hog_features)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.concatenate(img_features).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return bboxes


from movie import process_movie
from arrange_in_grid import arrange_in_grid

def pipeline(image, write_images=False, prefix='', frame=None):
    ystart = 400
    ystop = 656

    bboxes = []
    for scale in [1, 2]:
        bboxes.extend(find_cars(image, ystart, ystop, scale, svc, X_scaler,
                            p['orient'], p['pix_per_cell'], p['cell_per_block'],
                            p['spatial_size'], p['hist_bins'], p['spatial_feat'],
                            p['hist_feat'], p['hog_feat'], p['hog_channel']))


    #out_img = draw_bboxes(image, bboxes)
    out_img, heat = cars_from_bboxes(image, bboxes)

    maxheat = np.max(heat)
    heat = heat / maxheat * 255

    arranged = arrange_in_grid([
        out_img,
        cv2.cvtColor(heat.astype(np.uint8), cv2.COLOR_GRAY2RGB),
        ])

    return arranged

def main():
    movies = {
        'project': {
            'input': 'project_video.mp4',
            'output': 'project_video_output.mp4',
            'debug_folder': 'project_video_debug',
            'start_frame': 0,
            'end_frame': 5,
            'entire_clip': True,
            'debug_frames': [1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301],
        },
        'test': {
            'input': 'test_video.mp4',
            'output': 'test_video_output.mp4',
            'debug_folder': 'test_video_debug',
            'start_frame': 0,
            'end_frame': 5,
            'entire_clip': True,
            'debug_frames': [],
        },
    }

    import sys
    videos = sys.argv[1:]

    for video in videos:
        process_movie(movies[video], pipeline)

if __name__ == '__main__':
    main()
