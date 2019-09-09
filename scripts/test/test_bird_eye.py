#!/usr/bin/env python
import numpy as np
import cv2

import pdb

import two_d_guidance.trr.vision.utils as trr_vu


def work(img, intr_cam_calib_path, extr_cam_calib_path):
    cam = trr_vu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    undistorted_img = cam.undistort_img(img)
    p = trr_vu.ChristineBirdEyeParam()
    be = trr_vu.BirdEyeTransformer(cam, p)
    img_be = be.process(undistorted_img)
    cv2.imshow('bird eye', img_be)
    cv2.imwrite('/tmp/be_frame_000000.png', img_be)
    cv2.waitKey(0)
    

if __name__ == '__main__':

    intr_cam_calib_path = '/home/poine/.ros/camera_info/christine_camera_road_front.yaml'
    extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/christine_cam_road_front_extr.yaml'
    cam = trr_vu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    

    img_path = '/home/poine/work/robot_data/christine/vedrines_track/frame_000000.png'
    img =  cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    work(img, intr_cam_calib_path, extr_cam_calib_path)
