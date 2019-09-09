#!/usr/bin/env python
import numpy as np
import cv2

import pdb

import two_d_guidance.trr.vision.utils as trr_vu


if __name__ == '__main__':
    img_path = '/tmp/be_frame_000000.png'
    bgr_img =  cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(bgr_img.shape)
    thresholder = trr_vu.BinaryThresholder()
    thresh = thresholder.process_bgr(bgr_img)
    cv2.imshow('thresh', thresh)
    cv2.imshow('in', bgr_img)
    cv2.waitKey(0)
    
