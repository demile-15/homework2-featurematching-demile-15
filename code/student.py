import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import math
from helpers import compute_dino_feature_map, sample_dino_descriptors

'''
EXTRA CREDIT IMPORTS:

pip install pyflann-py3
from pyflann import FLANN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
'''

def plot_feature_points(image, x, y):
    '''
    Plot feature points for the input image. 
    
    Show the feature points given on the input image. Be sure to add the images you make to your writeup. 
    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of feature points
    :y: np array of y coordinates of feature points
    '''
    plt.imshow(image, cmap="gray")
    plt.scatter(x, y, alpha=0.9, s=3)
    plt.show()

def get_feature_points(image, window_width):
    '''
    Returns a set of feature points for the input image

    (Please note that we recommend implementing this function last and using cheat_feature_points()
    to test your implementation of get_feature_descriptors() and match_features())

    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional feature point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local window in pixels

    :returns:
    :xs: an np array of the x coordinates (column indices) of the feature points in the image
    :ys: an np array of the y coordinates (row indices) of the feature points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each feature point
    :scale: an np array indicating the scale of each feature point
    :orientation: an np array indicating the orientation of each feature point

    '''


    xgrads = gaussian_filter1d(image, 2, axis=0, order=1)
    ygrads = gaussian_filter1d(image, 2, axis=1, order=1)

    # constants
    alpha = 0.05
    min_distance = 8

    # square and multiply gradients
    ix2 = xgrads ** 2
    iy2 = ygrads ** 2
    ixy = np.multiply(xgrads, ygrads)


    '''
    EXTRA CREDIT: Detected interest points at multiple scales 

    An extra credit solution would have performed the gaussian filter with multiple values of sigma

    ie. gix2 = gaussian_filter(ix2, sigma) — where sigma is changed and they pick the sigma value that results in the best accuracy
    '''

    # perform gaussians
    gix2 = gaussian_filter(ix2, 2)
    giy2 = gaussian_filter(iy2, 2)
    gixy = gaussian_filter(ixy, 2)

    # get cornerness score
    c = np.multiply(gix2, giy2) - gixy ** 2 - alpha * (gix2 + giy2) ** 2

    # dynamic threshold
    threshold = ((np.sum(c) / c.shape[0] / c.shape[1]) * 99.5 + 0.5 * np.max(c)) / 100

    # threshold the cornerness
    thresholded = c < threshold
    c[thresholded] = 0

    # non-maxima suppression
    coords = feature.peak_local_max(c, min_distance)

    # get individual coordinate arrays
    xs = coords[:,1]
    ys = coords[:,0]

    # Exclude points too close to image boundary for descriptor extraction
    half = window_width // 2
    H, W = image.shape[:2]
    mask = (xs >= half) & (ys >= half) & (ys + half < H) & (xs + half < W)
    xs, ys = xs[mask], ys[mask]

    return xs, ys

'''
EXTRA CREDIT - Used Adaptive Non-Maximum Suprression:

def get_feature_points(image, window_width):
    sigma = 0.8
    alpha = 0.04 
    grad_x, grad_y = np.gradient(image)
    grad_x = grad_x ** 2
    grad_y = grad_y ** 2
    grad_xy = grad_x * grad_y
    
    I_x2, I_y2, I_xy = gaussian_filter(grad_x, sigma=sigma), gaussian_filter(grad_y, sigma=sigma), gaussian_filter(grad_xy, sigma=sigma)
    
    height, width = image.shape
    harris_cornerness = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            M = np.array([[I_x2[y, x], I_xy[y, x]], [I_xy[y, x], I_y2[y, x]]])
            
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            
            harris_cornerness[y, x] = det_M - alpha * (trace_M ** 2)
    min_distance = window_width // 2


    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    peaksCut = feature.peak_local_max(harris_cornerness, threshold_rel=0.0001, min_distance=min_distance)
    
    # xs = peaksCut[:,1]
    # ys = peaksCut[:,0]
    
    
    # Adaptive Non Maximal Suppression
    # step 1: sort peaks in descending order of response strength 
    
    sorted_peaks = sorted(peaksCut, key=lambda p: harris_cornerness[p[0], p[1]], reverse=True)
    supression_radii = [np.inf] * len(sorted_peaks)
    
    neighbor_radius = 500
    # going through the each of the peaks
    for index, (x,y) in enumerate(sorted_peaks):
        for _, (x2, y2) in enumerate(sorted_peaks):
            if x == x2 and y==y2:
                continue
            distance = np.sqrt((y2-y)**2 + (x2-x)*2)
            if distance > neighbor_radius:
                continue
            else:
                # get the suppression radii of all the 
                if harris_cornerness[x,y] - harris_cornerness[x2,y2] >= 0.1 * harris_cornerness[x,y]:
                    supression_radii[index] = min(supression_radii[index], distance)
    features = []
    for i, (y, x) in enumerate(sorted_peaks):
        if supression_radii[i] < np.inf:
            features.append((y, x, supression_radii[i]))
    
    top_n = 2300
    features = sorted(features, key=lambda f: f[2], reverse=True)[:min(int(len(features)),top_n)]
    
    xs = np.array([f[1] for f in features])
    ys = np.array([f[0] for f in features])

    return xs, ys
'''

# Takes in an x,y location and returns gradient at that point in (mag, dir) form
def gradient(g, x, y):
    xgd = g[x,y,0]
    ygd = g[x,y,1]
    mag = math.sqrt(xgd * xgd + ygd * ygd)
    dir = math.atan2(ygd, xgd) % (2 * math.pi)
    return (mag, dir)

# Takes in an x,y location and returns a descriptor
def sift_descriptor(image, x, y, feature_width, grads):
    '''
    EXTRA CREDIT: Different spatial layouts for your feature

    An example would be to have a 16-bin histogram rather than the usual 8-bin
    '''
    # initialize descriptor array
    descriptor = np.zeros(128)

    # initialize arrays for magnitude and direction of gradiants
    grad_array_mag = np.zeros((feature_width, feature_width))
    grad_array_dir = np.zeros((feature_width, feature_width))

    # store half feature width for readability
    half = int(feature_width / 2)

    # get magnitudes and directions for all pixels in feature
    for i in range(-half, half):
        for j in range(-half, half):
            g = gradient(grads, i + x, j + y)
            grad_array_mag[i + half, j + half] = g[0]
            grad_array_dir[i + half, j + half] = g[1]

    # loops for each sector of the feature
    for i in range(4):
        for j in range(4):

            # index of where to start this sector's bins in the descriptor array
            base_index = 8 * (4 * i + j)
            # loop over each pixel in the sector
            for k in range(int(feature_width / 4)):
                for l in range(int(feature_width / 4)):
                    # find which bin to put this gradient in
                    extra_index = int(grad_array_dir[i * 4 + k, j * 4 + l] / (2 * math.pi) * 8)   # floor to integer
                    # add the magnitude of this gradient to the bin
                    descriptor[base_index + extra_index] += grad_array_mag[i * 4 + k, j * 4 + l]

    # RootSIFT: L1-normalize then sqrt (Arandjelovic & Zisserman, 2012)
    l1 = np.sum(np.abs(descriptor))
    if l1 > 1e-8:
        descriptor = np.sqrt(descriptor / l1)
    return descriptor

def get_feature_descriptors(image, x_array, y_array, window_width, mode, image_file=None):
    '''
    Returns a set of feature descriptors for a given set of feature points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, normalize patches as your local feature descriptor. You will
    then need to implement the more effective SIFT-like feature descriptor.
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates (column indices) of feature points
    :y: np array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    :mode: "patch", "sift", or "dinov3". Switches between image patch
           descriptors and SIFT descriptors. ("dinov3" mode is handled by
           helpers.py — you don't need to implement it.)
    :image_file: (optional) path to the image file, used for DINOv3 cache lookup

    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. features[i] is the descriptor for
               point (x[i], y[i]), so the shape of features should be
               (len(x), feature dimensionality). For standard SIFT, feature
               dimensionality is 128. `Num points` may be less than len(x) if
               some points are rejected, e.g., if out of bounds.

    '''
    # DINOv3 is handled here — you don't need to implement it.
    if mode == "dinov3":
        cache_path = os.path.splitext(image_file)[0] + "_dinov3.npz" if image_file else None
        fmap, meta = compute_dino_feature_map(image, cache_path=cache_path)
        return sample_dino_descriptors(fmap, meta, x_array, y_array)

    # get gradients (for SIFT only)
    xgrads = gaussian_filter1d(image, 2, axis=0, order=1)
    ygrads = gaussian_filter1d(image, 2, axis=1, order=1)
    grads = np.stack([xgrads, ygrads], axis=2)

    half = window_width // 2
    features = []
    for i in range(len(x_array)):
        # get pixel coordinates of features
        xc = int(x_array[i])
        yc = int(y_array[i])

        if mode == "patch":
            # Cut out image patch
            patch = image[yc - half : yc + half, xc - half : xc + half]
            # Flatten
            vec = patch.flatten()
            # Normalize
            vec /= np.linalg.norm(vec)
            # Append to feature list
            features.append(vec)

        elif mode == "sift":
            # get descriptor and add to list
            desc = sift_descriptor(image, yc, xc, window_width, grads)
            # desc = image[yc - 8:yc + 8, xc - 8:xc + 8].reshape(-1)
            features.append(desc)

        '''
        EXTRA CREDIT: GLOH

        elif mode == "gloh":
        features = np.zeros((len(x_array), 384))
        sr_size = window_width // 4
        
        # gradient orientation
        grad_x = sobel_h(image)
        grad_y = sobel_v(image)
        ori = np.arctan2(grad_y, grad_x)
        
        for i in range(len(x_array)):
            # Subregions
            y = y_array[i]
            x = x_array[i]
            srs = []
            for r in range(4):
                for c in range(4):
                    sr = ori[y+r*sr_size:y+(r+1)*sr_size,
                                     x+c*sr_size:x+(c+1)*sr_size]
                    srs.append(sr)
            
            # GLOH descriptor for subregions
            descs = []
            for sr in srs:
                hist = np.zeros((24,))
                for angle in sr.flatten():
                    hist[int((angle / np.pi) * 12)] += 1

                descs.append(hist)

            feature = np.concatenate(descs)
            features[i,:] = feature / np.linalg.norm(feature)
        '''

    return np.asarray(features)

def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor in the other. 
    Implements the Nearest Neighbor Distance Ratio (NNDR) Test to help threshold
    and remove false matches.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Remember that the NNDR will return a number close to 1 for feature 
    points with similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides)

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - zip (python built in function)
        - np.argsort()

    :params:
    :im1_features: an np array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''

    # create empty matches array
    matches = np.zeros((im1_features.shape[0], 2))

    # Pairwise Euclidean distance matrix via expanded form:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b  (equivalent to cdist).
    dists = np.sqrt(np.sum(im1_features ** 2, 1).reshape(-1, 1) + np.sum(im2_features ** 2, 1).reshape(1, -1) - 2 * im1_features @ im2_features.T)
    
    # sort and argsort
    sorted_dist_indices = np.argsort(dists, axis=1)
    sorted_dists = np.sort(dists, axis=1)

    '''
    EXTRA CREDIT - KD TREE: 
    flann = FLANN()
    sorted_dists, dists = flann.nn(im2_features, im1_features, 
                                       2, algorithm="kmeans", branching=32, iterations=7, checks=16)
    '''
    # sorted for best matches from image 2 to 1
    sorted_dist_indices_transpose = np.argsort(dists, axis=0)

    # set matches to be from each feature in image 1 to the most similar feature in image 2
    matches[:, 0] = np.arange(im1_features.shape[0])
    matches[:, 1] = sorted_dist_indices[:, 0]

    # get ratio of best match similarity to next best match similarity
    ratios = sorted_dists[:, 0] / sorted_dists[:, 1]

    # discount matches that only go in one direction
    for i in range(matches.shape[0]):
        if sorted_dist_indices_transpose[0, int(matches[i, 1])] != i:
            ratios[i] *= 1.1

    # Optimal threshold determined from lecture slides
    optimal_threshold = 0.85

    # Threshold
    mask = (ratios < optimal_threshold)
    ratios = ratios[mask]
    matches = matches[mask]

    return matches

'''
EXTRA CREDIT: lower dimensional features
def get_lower_dimensional_features(features, dimensions=32):
    normalized_features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    normalized_features = imputer.fit_transform(normalized_features)
    pca = PCA(n_components=dimensions)
    pca.fit(normalized_features)
    lower_dimensional_features = pca.transform(normalized_features)
    return lower_dimensional_features
'''
