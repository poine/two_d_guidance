#!/usr/bin/env python
import math, numpy as np
import matplotlib.pyplot as plt
import cv2

import pdb

import two_d_guidance.trr.vision.utils as trr_vu

#
# compute bird eye map
#
# http://romsteady.blogspot.com/2015/07/calculate-opencv-warpperspective-map.html
#
def make_be_map(be, img):
    print()
    print('input img shape {}'.format(img.shape))
    print('output img shape {}'.format((be.h, be.w)))
    dest_w, dest_h = be.w, be.h

    xmap = np.zeros((dest_h, dest_w), np.float32)
    ymap = np.zeros((dest_h, dest_w), np.float32)

    Hinv = np.linalg.inv(be.H)
    M11, M12, M13 = Hinv[0,:]
    M21, M22, M23 = Hinv[1,:]
    M31, M32, M33 = Hinv[2,:]
    
    for y in range(dest_h):
        for x in range(dest_w):
            if 0:
                w = M31*float(x) + M32*float(y) + M33
                w = 1./w if w != 0. else 0.
                new_x = (M11 * float(x) + M12 * float(y) + M13) * w
                new_y = (M21 * float(x) + M22 * float(y) + M23) * w
                xmap[y,x], ymap[y,x]= new_x, new_y
            else:
                pt_be = np.array([[x], [y], [1]], dtype=np.float32)
                pt_imp = np.dot(Hinv, pt_be)
                pt_imp /= pt_imp[2]
                xmap[y,x], ymap[y,x] = pt_imp[:2]
                
    unwarped_img_mapped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    unwarped_img_full = cv2.warpPerspective(img, be.H, (be.w, be.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return unwarped_img_mapped, unwarped_img_full


#
# Compute undistortion map
#
# ./modules/calib3d/src/undistort.cpp
#
def make_undist_map(cam, img):
    dest_w, dest_h = cam.w, cam.h
    xmap = np.zeros((dest_h, dest_w), np.float32)
    ymap = np.zeros((dest_h, dest_w), np.float32)
    for y in range(dest_h):
        for x in range(dest_w):
            # xy are the coordinates in undistorted image
            pt_undist = np.array([[x], [y], [1]], dtype=np.float32)
            pt_imp = np.dot(cam.inv_undist_K, pt_undist)
            pt_imp /= pt_imp[2]
            pt_img = cv2.projectPoints(pt_imp.T, np.zeros(3), np.zeros(3), cam.K, cam.D)[0]
            xmap[y, x], ymap[y, x] = pt_img.squeeze()
    undistorted_img_mapped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    undistorted_img_full = cam.undistort_img(img)
    #u_ref = cv2.remap(img, cam.mapx, cam.mapy, cv2.INTER_LINEAR)
    cv2.imshow('undistorted mapped', undistorted_img_mapped)
    cv2.imshow('undistorted full', undistorted_img_full)
    cv2.waitKey(0)

    return undistorted_img_mapped, undistorted_img_full

#
# Compute combined undistortion + birdeye map
#    
# https://stackoverflow.com/questions/29944709/how-to-combine-two-remap-operations-into-one
#
def make_composite_map(be, cam, img):
    Hinv = np.linalg.inv(be.H)
    Kinv = cam.inv_undist_K
    dest_w, dest_h = be.w, be.h
    xmap = np.zeros((dest_h, dest_w), np.float32)
    ymap = np.zeros((dest_h, dest_w), np.float32)     
    for y in range(dest_h):
        for x in range(dest_w):
            pt_be = np.array([[x, y, 1]], dtype=np.float32).T
            pt_imp = np.dot(Kinv, np.dot(Hinv, pt_be))
            pt_imp /= pt_imp[2]
            pt_img = cv2.projectPoints(pt_imp.T, np.zeros(3), np.zeros(3), cam.K, cam.D)[0]
            xmap[y,x], ymap[y,x] = pt_img.squeeze()

    xmap_int, ymap_int = cv2.convertMaps(xmap, ymap, cv2.CV_16SC2)
    be_img_mapped2 = cv2.remap(img, xmap_int, ymap_int, cv2.INTER_LINEAR)
    #pdb.set_trace()
                    
    be_img_mapped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    be_img = cv2.warpPerspective(cam.undistort_img(img), be.H, (be.w, be.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imshow('bird eye mapped', be_img_mapped)
    cv2.imshow('bird eye mapped2', be_img_mapped2)
    cv2.imshow('bird eye full', be_img)
    cv2.waitKey(0)




def work(img, intr_cam_calib_path, extr_cam_calib_path):
    cam = trr_vu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    undistorted_img = cam.undistort_img(img)
    p = trr_vu.ChristineBirdEyeParam()
    be = trr_vu.BirdEyeTransformer(cam, p)

    if 0: # OK, sortof... needs to investigate the differences - interpolation?
        img_be, img_be_ref = make_be_map(be, undistorted_img)
        print('same? {}'.format(np.allclose(img_be, img_be_ref)))
        cv2.imshow('bird eye', img_be)
        cv2.imshow('bird eye ref', img_be_ref)
        cv2.imwrite('/tmp/be_frame_000000.png', img_be)
        cv2.waitKey(0)
    if 0: # OK
        img_undist, img_undist_ref = make_undist_map(cam, img)
    if 1:
        make_composite_map(be, cam, img)
    

    
    

if __name__ == '__main__':

    intr_cam_calib_path = '/home/poine/.ros/camera_info/christine_camera_road_front.yaml'
    extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/christine_cam_road_front_extr.yaml'
    cam = trr_vu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    

    img_path = '/home/poine/work/robot_data/christine/vedrines_track/frame_000000.png'
    img =  cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    work(img, intr_cam_calib_path, extr_cam_calib_path)
