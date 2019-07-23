#!/usr/bin/env python
import numpy as np, cv2
import matplotlib.pyplot as plt
import timeit

import pdb

import two_d_guidance.trr_vision_utils as trvu


if __name__ == '__main__':
    intr_cam_calib_path = '/home/poine/.ros/camera_info/camera_road_front.yaml'
    extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/caroline_cam_road_front_extr.yaml'
    cam = trvu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    img_path = '/home/poine/work/robot_data/caroline/line_z_01.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    number=100
    def test1(): cam.undistort_img(img)
    res1 = timeit.timeit('test1()', setup='from __main__ import test1', number=number) 
    print('test1 {} ({}s/loop)'.format(res1, res1/number))

    img_undistorted = cam.undistort_img(img) # costly (0.05s)


    #m1type = cv2.CV_32FC1
    m1type = cv2.CV_16SC2
    mapx, mapy = cv2.initUndistortRectifyMap(cam.K, cam.D, None, cam.undist_K, (cam.w, cam.h), m1type)
    def test2(): cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    res2 = timeit.timeit('test2()', setup='from __main__ import test2', number=number) 
    print('test2 {} ({}s/loop)'.format(res2, res2/number))
    
    img_undistorted2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray_undistorted = cam.undistort_img(img_gray)
    
    
    cv2.imshow('orig', img)
    cv2.imshow('undistorted', img_undistorted)
    cv2.imshow('undistorted2', img_undistorted2)
    #cv2.imshow('gray undistorted', img_gray_undistorted)
    cv2.waitKey(0)
    
