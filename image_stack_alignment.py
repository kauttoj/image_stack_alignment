import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import largestinteriorrectangle as lir
import copy
import matplotlib.patches as patches

# This script performs alignment of images using openCV
# Input: Folder of images. They should be same size with enough similar content
# Output: Three folders of images in the original image folder:
#    out_align = aligned images which might contain white borders corresponding shifts
#    out_align_crop = aligned images with white borders cropped
#    out_align_crop_resize_X = aligned and cropped images with desired output resolution, additional cropping if aspect ratio changes (X=center, top, bottom, left or right)

# Code contains some segments found from internet, e.g., stack overflow (thanks for corresponding authors)

# JanneK, 12.3.2023

# input params
input_path = r'C:\Users\janne\Downloads\kuvatblob\sohva' # take all images here
reference_img_name = None # which image to use as reference. Should be the one with least correction needed and highest quality! If none, we pick the one in the middle
final_size = (1920,1080) # final size in pixels of aligned and cropped images (width, height)
image_extensions = ('jpg','png','tiff','jpeg','bmp') # allowed image extensions, omit others
crop_position = 'bottom' # how to position final cropped image to retain more certain part of image, options are 'center','top','bottom','left','right'

# initializations
input_path = input_path + '\\'
output_path = input_path + 'out' # will be created if doesn't exist
blur_pixels = (3,3) # smoothing kernel size, helps in alignment
crop_factor = 2 # downsample image to speed-up computing optimal crop window (by largestinteriorrectangle)

assert crop_position in ['center','top','bottom','left','right'],'crop_position must be one of "center, top, bottom, left, right"'

# Iterate directory
images = []
for path in os.listdir(input_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(input_path, path)):
        if os.path.splitext(path)[1][1:] in image_extensions:
            images.append(path)

# create output directories
for folder in ['_align','_align_crop','_align_crop_resize_' + crop_position]:
    if not os.path.isdir(output_path + folder):
        os.mkdir(output_path + folder)

# pick reference
if reference_img_name is None:
    reference_img_name = images[int(len(images)/2)]
    print('reference not set, using middle ("%s")' % reference_img_name)

# read reference image to be used in alignment
img2_reference = plt.imread(input_path + reference_img_name)  # Reference image.
img2_reference = cv2.blur(img2_reference,blur_pixels) # smooth
img2_reference = cv2.cvtColor(img2_reference, cv2.COLOR_BGR2GRAY)
height, width = img2_reference.shape

# Create ORB detector with 7000 features.
orb_detector = cv2.ORB_create(7000)
kp2, d2 = orb_detector.detectAndCompute(img2_reference, None)

# omit downsampling for small images (assume reference represents typical size)
if max(img2_reference.shape)<3000:
    crop_factor = 1

transformed_imgs = []
rect_global  = None
areas = []

total_k = len(images)
print('Processing...')
for k,image_to_align in enumerate(images):

    img1_color = plt.imread(input_path + image_to_align) # read image to align
    img1_color_orig = copy.deepcopy(img1_color) # make copy
    img1_color = cv2.blur(img1_color,blur_pixels) # blur

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color_orig,homography,(width, height),borderMode=cv2.BORDER_CONSTANT,borderValue=(255, 255, 255))
    # Save the output.

    # check if have same pixel count
    if img1_color_orig.shape != transformed_img.shape:
        print('warning: image size changed in transformation (try to pick largest reference)')

    # create empty image
    img_1 = np.zeros([height,width, 1], dtype=float)
    img_1.fill(1.0)

    # warp
    transformed_img0 = cv2.warpPerspective(img_1,homography,(width, height))
    # downscale
    transformed_img0 = cv2.resize(transformed_img0,(int(width / crop_factor),int(height / crop_factor)))
    transformed_img0 = transformed_img0>0.90 # threshold back to binary

    # find maximal bounding box (with speedup trick)
    cv_grid = transformed_img0.astype("uint8") * 255
    contours, _ = cv2.findContours(cv_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = contours[0][:, 0, :]
    rect = lir.lir(transformed_img0, contour) # x, y, width, height

    # find maximal bounding box
    #rect = lir.lir(transformed_img0) # x, y, width, height

    if 0: # plotting for debugging
        plt.close('all')
        fig, ax = plt.subplots(1,3,figsize=(25,5))
        plt.sca(ax[0])
        plt.imshow(img1_color_orig)
        plt.sca(ax[1])
        plt.imshow(transformed_img)
        img = transformed_img0.astype(np.uint8) * 255
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(0, 255, 0),4)
        plt.sca(ax[2])
        plt.imshow(img)  # Resize by width OR
        plt.show()

    # upscale coordinates
    rect = rect*crop_factor
    rect_orig = tuple(rect) # copy

    # convert to x-y coordinates
    rect = (rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3])

    area = (rect[3] - rect[1]) * (rect[2] - rect[0]) # area in pixels
    areas.append(area)

    # update global bounding box
    if rect_global is None:
        rect_global = rect
    else:
        rect_global = (max(rect[0],rect_global[0]),max(rect[1],rect_global[1]),min(rect[2],rect_global[2]),min(rect[3],rect_global[3]))

    if 0: # plotting for debugging
        plt.close('all')
        img = cv2.rectangle(transformed_img,(rect_global[1],rect_global[0]),(rect_global[2],rect_global[3]),(0, 255, 0),4)
        plt.imshow(img)  # Resize by width OR
        plt.show()

    # save aligned image
    plt.imsave(output_path  + '_align' + '\\' + image_to_align, transformed_img)

    # add to list
    transformed_imgs.append(transformed_img)

    # print some stats
    area = (rect_global[3]-rect_global[1])*(rect_global[2]-rect_global[0])
    print('...image %i/%i ("%s"), global bounding box: x = %i - %i, y = %i - %i (area %i)' % (k+1,total_k,image_to_align,rect_global[0],rect_global[2],rect_global[1],rect_global[3],area))

areas = np.array(areas)
mean_area = np.median(areas)
percentage = 100*areas/mean_area

# function to crop and resize to specific size, additionallu choose location to retain more from certain part of the image
def crop_and_resize(img, w, h,crop_position='center'):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w / h
    input_aspect_ratio = im_w / im_h
    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio * h)
        im_h_r = h
        img = cv2.resize(img, (im_w_r, im_h_r))
        x1 = int((im_w_r - w) / 2)
        x2 = x1 + w
        if crop_position == 'center':
            img = img[:, x1:x2, :]
        elif crop_position == 'left':
            img = img[:,0:w, :]
        elif crop_position == 'right':
            img = img[:,-w:, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w / input_aspect_ratio)
        img = cv2.resize(img, (im_w_r, im_h_r))
        y1 = int((im_h_r - h) / 2)
        y2 = y1 + h
        if crop_position == 'center':
            img = img[y1:y2, :, :]
        elif crop_position == 'top':
            img = img[0:h, :, :]
        elif crop_position == 'bottom':
            img = img[-h:, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv2.resize(img, (w, h))

    return img

print('\nCropping...')
for k,image_to_align in enumerate(images):
    # apply global bounding box
    transformed_img = transformed_imgs[k][rect_global[1]:rect_global[3],rect_global[0]:rect_global[2], :]
    plt.imsave(output_path + '_align_crop' + '\\' + image_to_align, transformed_img,dpi=1500)

    # convert to desired shape
    transformed_img = crop_and_resize(transformed_img,final_size[0],final_size[1],crop_position)
    plt.imsave(output_path + '_align_crop_resize_' + crop_position + '\\' + image_to_align, transformed_img, dpi=1500)

    print('...image %i ("%s"), area size %.1f%% of median' % (k + 1,image_to_align,percentage[k]))

print('\nAll done!')