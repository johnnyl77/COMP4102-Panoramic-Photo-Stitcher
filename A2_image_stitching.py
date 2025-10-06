import os
import cv2
import numpy as np
from argparse import ArgumentParser

def load_image(path, size=None):
  """
  Load an image from a file path (using OpenCV), keep the color channels, and resize it if needed

  Args:
    path: str, path to the image file

  Returns:
    image: np.array, the color image as a numpy array (H x W x C)
  """

  image = cv2.imread(path)

  # If image is valid then it returns it
  if image is not None:
    if size is not None:
      image = cv2.resize(image, size)
    return image
  else:
    print("Cannot load image!")
  
  return None

def extract_features(image):
  """
  Extract features from an image using a feature detector (e.g. SIFT)

  see: 
    - https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
    - https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

  Args:
    image: np.array, the image as a numpy array

  Returns:
    (keypoints, features): tuple, the keypoints and features 
  """
  
  # Convert to grayscale img first for sift
  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # create a SIFT detector
  sift = cv2.SIFT_create()

  # detect and compute the keypoints and features
  kps, features = sift.detectAndCompute(grayscale_image, None)

  # return the keypoints and features
  return kps, features

def match_features(features1, features2, lowe_ratio=0.7, k=2, flann_algorithm_index=1, flann_trees=5, flann_checks=50):
  """
  Match features between two images using a feature matcher (e.g. FLANN)

  see:
    - https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    - https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

  Args:
    features1: np.array, the features from the first image
    features2: np.array, the features from the second image
    lowe_ratio: float, the Lowe's ratio test multiplier
    k: int, the number of nearest neighbors to find
    flann_algorithm_index: int, the algorithm index for the FLANN matcher
    flann_trees: int, the number of trees for the FLANN matcher
    flann_checks: int, the number of checks for the FLANN matcher

  Returns:
    matches: np.array, the good matches as a numpy array
  """
  
  # create a FLANN matcher
  index_params = dict(algorithm=flann_algorithm_index, trees=flann_trees)
  search_params = dict(checks=flann_checks)
  flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

  # find the k best matches (using knnMatch)
  matches = flann.knnMatch(features1, features2, k=k)

  # filter the matches using the Lowe's ratio test
  good_matches = []

  for m,n in matches:
    if m.distance < lowe_ratio * n.distance:
      good_matches.append(m)

  # return the good matches
  return good_matches

def estimate_homography(src_features, dest_features, ransac_threshold):
  """
  Estimate a homography matrix from matched features using RANSAC
  
  see:
    - https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

  Args:
    src_features: np.array, the features from the source image
    dest_features: np.array, the features from the destination image
    ransac_threshold: float, the RANSAC threshold

  Returns:
    homography: np.array, the homography matrix as a numpy array
  """
  
  # convert the features to numpy arrays
  src_pts = np.float32([f.pt for f in src_features]).reshape(-1,1,2)
  dest_pts = np.float32([f.pt for f in dest_features]).reshape(-1,1,2)

  # estimate the homography matrix using RANSAC
  M, mask = cv2.findHomography(srcPoints=src_pts, dstPoints=dest_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)

  # return the homography matrix
  return M, mask

def warp_image(image, homography, size=None):
  """
  Warp an image using a homography matrix (perspective transformation)

  Args:
    image: np.array, the image as a numpy array
    homography: np.array, the homography matrix as a numpy array

  Returns:
    warped_image: np.array, the warped image as a numpy array
  """
  
  # warp the image using the homography matrix
  if size is not None:
    warped_image = cv2.warpPerspective(image, homography, size)
  else:
    warped_image = cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]))
  
  # return the warped image
  return warped_image

def stitch_images(opt):
  """
  Stitch the images together using feature matching and homography estimation.
  (Warp the right image and stitch it to the left image)
  Blend the images together to create a seamless transition.

  Args:
    opt: argparse.Namespace, the command line arguments

  Returns:
    stitched_image: np.array, the stitched image as a numpy array
  """
  
  # Load the left and right images and resize them
  left_image = cv2.imread(opt.left_image)
  right_image = cv2.imread(opt.right_image)
  left_image = cv2.resize(left_image, (opt.width, opt.height))    
  right_image = cv2.resize(right_image, (opt.width, opt.height))

  # extract features from the images
  left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
  right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

  # match the features between the images
  sift = cv2.SIFT_create()

  # extract the matched features
  left_kps, left_features = sift.detectAndCompute(left_gray, None)
  right_kps, right_features = sift.detectAndCompute(right_gray, None)

  index_params = dict(algorithm=opt.flann_algorithm_index, trees=opt.flann_trees)
  search_params = dict(checks=opt.flann_checks)
  flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

  # find the k best matches (using knnMatch)
  matches = flann.knnMatch(left_features, right_features, k=opt.k)

  # filter the matches using the Lowe's ratio test
  good_matches = []

  for m,n in matches:
    if m.distance < opt.lowe_ratio * n.distance:
      good_matches.append(m)
  
  src_pts = np.float32([left_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
  dest_pts = np.float32([right_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

  # estimate the homography matrix (Swap dest and src points here because we are warping the right and using left img as reference)
  M, mask = cv2.findHomography(srcPoints=dest_pts, dstPoints=src_pts, method=cv2.RANSAC, ransacReprojThreshold=opt.ransac_threshold)

  # calculate the size of the two images stitched together
  h1, w1 = left_image.shape[0], left_image.shape[1]
  h2,w2 = right_image.shape[0], right_image.shape[1]
  h3 = max(h1,h2)
  w3 = w1+w2

  # Warp the right image using the homography matrix
  warped_right_image = cv2.warpPerspective(right_image, M, (w3, h3))

  # Stitch the images together (array slicing)
  stitched_image = warped_right_image.copy()
  stitched_image[0:h1, 0:w1] = left_image

  # return the stitched image
  return stitched_image


def main():

  parser = ArgumentParser()
  parser.add_argument('--output', type=str, default='output/stitched_image.jpg', help='The output file path')
  parser.add_argument('--left_image', type=str, default='data/img_left.jpg', help='The left image file path')
  parser.add_argument('--right_image', type=str, default='data/img_right.jpg', help='The right image file path')
  parser.add_argument('--width', type=int, default=720, help='Resize the images to this width (faster computation)')
  parser.add_argument('--height', type=int, default=480, help='Resize the images to this height (faster computation)')
  parser.add_argument('--lowe_ratio', type=float, default=0.7, help='The Lowe\'s ratio test multiplier')
  parser.add_argument('--k', type=int, default=2, help='The number of nearest neighbors to find in the FLANN knnMatch')
  parser.add_argument('--flann_algorithm_index', type=int, default=1, help='The algorithm index for the FLANN matcher')
  parser.add_argument('--flann_trees', type=int, default=5, help='The number of trees for the FLANN matcher')
  parser.add_argument('--flann_checks', type=int, default=50, help='The number of checks for the FLANN matcher')
  parser.add_argument('--ransac_threshold', type=float, default=5.0, help='The RANSAC threshold')

  opt = parser.parse_args()

  # Create the output directory
  os.makedirs(os.path.dirname(opt.output), exist_ok=True)

  # stitch the images together
  stitched_image = stitch_images(opt)

  # write the stitched image to a file
  cv2.imwrite(opt.output, stitched_image)

if __name__ == '__main__':
  main()