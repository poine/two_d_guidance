#!/usr/bin/env python
import numpy as np
import cv2


import two_d_guidance.trr_vision_utils as trrvu

        

        
def work(img):
    #img  = np.flip(imgaxis=1)
    ssd = trrvu.StartFinishDetector()
    ssd.process_image(img)
    
  
    cv2.imshow('orig', img)
    cv2.imshow('red mask', ssd.red_mask)
    cv2.imshow('green mask', ssd.green_mask)
    img2 = ssd.draw(img)
    cv2.imshow('contours', img2)
    #ssd.contour_red.draw(img)
    #ssd.contour_green.draw(img)
    cv2.waitKey(0)

if __name__ == '__main__':
    img =  cv2.imread('/home/poine/work/robot_data/caroline/gazebo/start_line_05.png', cv2.IMREAD_COLOR)
    
    work(img)
