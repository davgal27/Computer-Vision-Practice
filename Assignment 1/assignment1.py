# coding: utf-8

import numpy as np
import cv2

# In this assignment, you should learn the very basics of computer vision.
# How to read and write images and videos, basic access to image pixels
# and how to perform simple color manipulations. 

# You should write vectorized code using numpy and OpenCV. 
# You don't have to (and should not) use any "for" cycles.

# IMPORTANT: Fill in blank places marked with ## FILL.
# IMPORTANT: Do not change the rest of the code.
# IMPORTANT: Submit only this single file.
# IMPORTANT: You can run this file as: python du01.py -i <path_to_image_file> -v <path_to_video_file>

# This should help with the assignment (especially slicing):
# * Indexing numpy arrays http://scipy-cookbook.readthedocs.io/items/Indexing.html


# Prepare and load python environment on linux.
# Python should already be installed. 
# Install virtualenv by:
# $ sudo apt install virtualenv
#
# Run following commands:
# $ virtualenv env
# $ source env/bin/activate
# $ pip install numpy opencv-python
# $ python du01-reference.py -i image.jpg -v video.mp4


def parseArguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='Input video file name.')
    parser.add_argument('-i', '--image', help='Input image file name.')
    args = parser.parse_args()
    return args


def image(imageFileName):
    # read image
    img = cv2.imread(imageFileName)## FILL 
    if img is None:
        print(f"Error: Unable to read image file {imageFileName}.")
        exit(-1)

    # print image width, height, and channel count
    print("Image dimensions:", "\nwidth:", img.shape[1], "\nheight:", img.shape[0], "\ncolour channels:", img.shape[2]) ## FILL
    
    # Resize to width 400 and height 500 with bicubic interpolation.
    img = cv2.resize(img, (400, 500), interpolation=cv2.INTER_CUBIC) ## FILL
          
    # Print mean image color and standard deviation of each color channel
    (mean, stddev) = cv2.meanStdDev(img)
    mean = mean.flatten()
    stddev = stddev.flatten()
    print('\nImage mean and standard deviation:', "\nBlue mean:", mean[0], "\nGreen mean:", mean[1], "\nRed mean:", mean[2],
    "\nBlue Sd:", stddev[0], "\nGreen Sd:", stddev[1], "\nRed Sd:", stddev[2])## FILL
    


    # Create a copy of the image and 
    # fill horizontal rectangle with gray color with intesity 128 (#808080).
    # Position x1=50,y1=120 and size width=200, height=50
    # You can use OpenCV function or use slicing
    ## FILL
    rectimg = img.copy()
    cv2.rectangle(rectimg, (50,120), (250,170), (128,128,128), -1)
    # write result to file
    cv2.imwrite('rectangle.png', rectimg)


    
    # Create a copy of the original image and 
    # fill every third pixel column in the top half of the image with black color.
    # The first pixel column should be black.  
    # The rectangle from previous step should not be visible.
    # The lines can be drawn by slicing the correct pixels from img and writing the correct value. 
    ## FILL
    stripedimg = img.copy()
    tophalf = stripedimg[0:250, :]
    tophalf[:, ::3] = (0,0,0) 

    # write result to file
    cv2.imwrite('striped.png', stripedimg)


    
    # Create a copy of the original image and 
    # set all pixels with any collor value lower than 100 to black (all color channels will be black).
    # Ideal solution uses numpy features such as slicing, np.where and ".any()".
    ## FILL     
    clippedimg = img.copy()
    mask = cv2.inRange(clippedimg, (0,0,0), (99,99,99))
    clippedimg[mask > 0] = (0,0,0)

    # write result to file
    cv2.imwrite('clip.png', clippedimg)


def video(videoFileName):
    # open video file and get basic information
    videoCapture = cv2.VideoCapture(videoFileName)    
    frameRate = videoCapture.get(cv2.CAP_PROP_FPS)## FILL
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))## FILL
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not videoCapture.isOpened():
        print(f"Error: Unable to open video file for reading {videoFileName}.")
        exit(-1)
    
    # open video file for writing
    videoWriter  = cv2.VideoWriter(
        'videoOut.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 
        frameRate, (frame_width, frame_height))        
    if not videoWriter.isOpened():
        print(f"Error: Unable to open video file for writing {videoFileName}.")
        exit(-1)
                
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break;
        
        # Flip image upside down. Numpy slicing is the way to go.
        ## FILL
        flipped_vid = frame[::-1, :, :]

        

        # The following section can be tricky. 
        # The input image has unsigned 8-bit values (np.uint8) 
        # and some of the steps may require the image to have values 
        # in the range 0-255 or 0-1 depending on how you write the code. 
        # Think about the data types uint8/float, value cliping and 
        # possible overflows.
        
        # Add white noise (normal distribution).
        # Standard deviation should be 5.
        # use np.random
        # Think about data types and exeding value range
        ## FILL  
        noise = np.random.normal(0, 5, flipped_vid.shape)
        frame_with_noise = flipped_vid.astype(np.float32) + noise
        frame_with_noise = np.clip(frame_with_noise, 0, 255)
        frame_with_noise = frame_with_noise.astype(np.uint8)


        # Add gamma correction.
        # y = x^1.2 -- the image to the power of 1.2
        # Think what hapens for negative input values? Should you have negative values?
        ## FILL  
        gamma = 1.2
        frame_gamma_correct = frame_with_noise.astype(np.float32) / 255
        frame_gamma_correct = np.power(frame_gamma_correct, gamma)
        frame_gamma_correct = np.clip(frame_gamma_correct * 255 , 0, 255)
        frame_gamma_correct = frame_gamma_correct.astype(np.uint8)



        # Dim blue color to half intensity.
        ## FILL  
        frame_gamma_correct[:, :, 0] = (frame_gamma_correct[:, :, 0] * 0.5).astype(np.uint8)
                
        # Invert colors. I would do it in numpy, but OpenCV has function for it as well.
        ## FILL  
        inverted_frame_final = 255 - frame_gamma_correct

        
        # Display the processed frame.         
        cv2.imshow("Output", inverted_frame_final)
        # Write the resulting frame to video file.
        videoWriter.write(inverted_frame_final)   
            
        # End the processing on pressing Escape.
        ## FILL  
        if cv2.waitKey(10) == 27:
            break
        
    cv2.destroyAllWindows()        
    videoCapture.release()
    videoWriter.release()          
    
def main():
    args = parseArguments()
    np.random.seed(1)
    image(args.image)
    video(args.video)


if __name__ == "__main__":
    main()
