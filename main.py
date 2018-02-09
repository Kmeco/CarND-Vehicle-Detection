import glob
import time
import pickle
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from helper_functions import *

# Define parameters for feature extraction
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (12, 12) # Spatial binning dimensions
hist_bins = 18    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

with open('classifier.pkl','rb') as f:
    svc = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    X_scaler = pickle.load(f)
print('loaded')
def process_image(image):
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 656],
                           xy_window=(256, 256), xy_overlap=(0.75, 0))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 592],
                            xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 528],
                            xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    hot_windows = []
    hot_windows += search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                  orient=orient, pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block,
                                  hog_channel=hog_channel, spatial_feat=spatial_feat,
                                  hist_feat=hist_feat, hog_feat=hog_feat)

    return draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

white_output = 'results/test_video_out_3.mp4'
clip1 = VideoFileClip("test_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)



