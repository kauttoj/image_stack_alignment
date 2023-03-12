# image_stack_alignment
This script performs alignment of image stacks using openCV. It will read images from folder and save results in other folders.
Typical use case is motion correction. Can be applied to video frames, but the process is slow for hundreds of images or more (there are better alternatives for videos).

Input: Folder of images. They should be same size with enough similar content  
Output: Three folders of images in the original image folder:  
-out_align = aligned images which might contain white borders corresponding shifts  
-out_align_crop = aligned images with white borders cropped  
-out_align_crop_resize_X = aligned and cropped images with desired output resolution, additional cropping if aspect ratio changes (X=center, top, bottom, left or right)  
    
In order for alignment to work, images must have enough similar content.

12.3.2023
JanneK
