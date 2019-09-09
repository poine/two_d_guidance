#!/usr/bin/env python
import math, numpy as np
import cv2

import pdb

import two_d_guidance.trr.vision.utils as trr_vu

#
# http://romsteady.blogspot.com/2015/07/calculate-opencv-warpperspective-map.html
#
def make_map(be, img):
    print()
    print('input img shape {}'.format(img.shape))
    print('output img shape {}'.format((be.h, be.w)))
    dest_w, dest_h = be.w, be.h

    xmap = np.zeros((dest_h, dest_w), np.float32)
    ymap = np.zeros((dest_h, dest_w), np.float32)

    H = np.linalg.inv(be.H)
    M11, M12, M13 = H[0,:]
    M21, M22, M23 = H[1,:]
    M31, M32, M33 = H[2,:]
    
    for y in range(dest_h):
        for x in range(dest_w):
            w = M31*float(x) + M32*float(y) + M33
            w = 1./w if w != 0. else 0.
            new_x = (M11 * float(x) + M12 * float(y) + M13) * w
            new_y = (M21 * float(x) + M22 * float(y) + M23) * w
            xmap[y,x] = new_x#min(x, img.shape[1])
            ymap[y,x] = new_y#min(y, img.shape[0])
    

    unwarped_img = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    print('output img shape {}'.format(unwarped_img.shape))
    #unwarped_img = cv2.warpPerspective(img, be.H, (be.w, be.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return unwarped_img


def work(img, intr_cam_calib_path, extr_cam_calib_path):
    cam = trr_vu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    undistorted_img = cam.undistort_img(img)
    p = trr_vu.ChristineBirdEyeParam()
    be = trr_vu.BirdEyeTransformer(cam, p)

    #img_be = be.process(undistorted_img)
    img_be = make_map(be, undistorted_img)
    

    
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
