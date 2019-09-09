#!/usr/bin/env python
import numpy as np
import cv2

import pdb

import smocap
import fl_utils as flu
import fl_vision_utils as flvu

def plot_polygon_on_img(img, cam, poly_ref, closed=True, color=(0,128,0), line_width=2):
    poly_img = cam.project(poly_ref).squeeze().astype(int)
    #pdb.set_trace()
    for p1,p2 in zip(poly_img, poly_img[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color, line_width)
    if closed:
        cv2.line(img, tuple(poly_img[-1]), tuple(poly_img[0]), color, line_width)

def plot_polygon_on_undistorted_img(img, cam, poly_ref, color=(0,128,0), line_width=2):
    poly_img = cam.project(poly_ref)
    poly_img_undistorted = [tuple(p) for p in cam.undistort_points(poly_img).astype(int).squeeze()]
    for p1,p2 in zip(poly_img_undistorted, poly_img_undistorted[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color, line_width)

    c = (0,255,0)
    print poly_img_undistorted
    for t,p in zip(('1bl', '2tl', '3tr', '4br'), poly_img_undistorted):
        cv2.circle(img, p , 2, c, -1); cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, 1., c, 1)

def work(img, intr_cam_calib_path, extr_cam_calib_path):
    cam = flvu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)
    #cam = smocap.Camera(0, 'cam0')
    #intr_cam_calib_path = '/home/poine/.ros/camera_info/ueye_drone.yaml'
    #cam.load_intrinsics(intr_cam_calib_path)
    #cam.set_undistortion_param(alpha=1.)
    

    undistorted_img = cam.undistort_img(img)
    # base_footprint to camera optical_frame
    # if 0:
    #     bf_to_camo_t = [ 0.00039367,  0.3255541,  -0.04368782]
    #     bf_to_camo_q = [ 0.6194402,  -0.61317113,  0.34761617,  0.34565589]
    #     cam.set_location(bf_to_camo_t, bf_to_camo_q)
    # else:
    #     #extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/pierrette_cam1_extr.yaml'
    #     cam.load_extrinsics(extr_cam_calib_path)

    #p = flvu.BirdEyeParam(x0=0.2, dx=1.2, dy=0.6, w=752)
    #p = flvu.BirdEyeParam(x0=0.2, dx=0.6, dy=0.6, w=752)
    p = flvu.CarolineBirdEyeParam()
    be = flvu.BirdEyeTransformer(cam, p)
    img_be = be.process(undistorted_img)
    be.plot_calibration(undistorted_img)
    if 0:
        rh_side = np.array([(0.9, 0., 0), (0.9, 0.3, 0), (0.9, 0.6, 0.), (1.5, 0., 0.), (2.7, 0.6, 0.)])
        lh_side = np.array(rh_side[::-1]); lh_side[:,1] = -lh_side[:,1]
        poly_ref = np.append(rh_side, lh_side, axis=0)
    plot_polygon_on_img(img, cam, be.param.va_bf)
    plot_polygon_on_undistorted_img(undistorted_img, cam, be.param.va_bf)
    cv2.imshow('orig', img)
    cv2.imshow('undistorted', undistorted_img)
    cv2.imshow('bird eye', img_be)
    cv2.imwrite('/home/poine/work/robot_data/jeanmarie/be_z_room_line_11.png', img_be)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    #img_path = '/home/poine/work/robot_data/jeanmarie/floor_tiles_03.png'
    #img_path = '/home/poine/work/robot_data/jeanmarie/z_room_line_11.png'
    #intr_cam_calib_path = '/home/poine/.ros/camera_info/ueye_drone.yaml'
    #extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/pierrette_cam1_extr.yaml'

    #img_path = '/home/poine/work/robot_data/caroline/floor_tiles_z_01.png'
    img_path = '/home/poine/work/robot_data/caroline/line_z_01.png'
    intr_cam_calib_path = '/home/poine/.ros/camera_info/camera_road_front.yaml'
    extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/caroline_cam_road_front_extr.yaml'
     
    img =  cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    work(img, intr_cam_calib_path, extr_cam_calib_path)
