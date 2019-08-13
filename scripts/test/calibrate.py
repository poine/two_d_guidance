#!/usr/bin/env python

import logging, glob, yaml
import numpy as np, cv2
import rospy, sensor_msgs.msg, tf.transformations
import pdb

import smocap
import two_d_guidance.trr_vision_utils as trrvu
####
#### started from smocap calibrate_camera_extrinsic
#### I should have fixed it instead...
####
LOG = logging.getLogger('fl_calibrate')

def test_1():
    img = cv2.imread('/home/poine/work/robot_data/jeanmarie/chessboard_01.png',  cv2.IMREAD_UNCHANGED)

    # compute chessboard corners in chessboard frame
    cb_geom, cb_size = (8,6), 0.108
    cb_points = np.zeros((cb_geom[0]*cb_geom[1], 3), np.float32)
    cb_points[:,:2] = cb_size * np.mgrid[0:cb_geom[0],0:cb_geom[1]].T.reshape(-1,2)
    # detect the marker
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS|cv2.CALIB_CB_SYMMETRIC_GRID
    ret, corners = cv2.findChessboardCorners(img, cb_geom, flags=flags)
    
    print('marker detection {}'.format('succeeded' if ret else 'failed'))
    
    # draw the detected marker
    cv2.drawChessboardCorners(img, cb_geom, corners, ret)

    cv2.imshow('my image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate_intrinsic(_dir='/tmp/cam_calib', _prefix='left', cb_geom=(8,6), cb_size=0.025, refine_corners=True):

    img_glob = '{}/{}*'.format(_dir, _prefix)
    LOG.info(" loading images: {}".format(img_glob))
    img_path = glob.glob(img_glob)
    img_path.sort()
    LOG.info(" found {} images".format(len(img_path)))
    img_points, object_points = [], [] # 2d points in image plane, 3d point in real world space
    cb_points = np.zeros((cb_geom[0]*cb_geom[1], 3), np.float32)
    cb_points[:,:2] = cb_size*np.mgrid[0:cb_geom[0],0:cb_geom[1]].T.reshape(-1,2)
    for p in img_path:
        img = cv2.imread(p)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS|cv2.CALIB_CB_SYMMETRIC_GRID
        ret, corners = cv2.findChessboardCorners(img_gray, cb_geom, flags=flags)
        if ret:
            object_points.append(cb_points)
            if not refine_corners:
                img_points.append(corners)
            else:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(img_gray,corners,(11,11),(-1,-1),criteria)
                img_points.append(corners2)
            
    LOG.info(" successfully extracted {} chessboard pattern in {} images".format(cb_geom, len(img_points)))
    img_shape = img_gray.shape[::-1]
    rep_err, cmtx, distk, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, img_shape,None,None)
    LOG.info(" cv calibration:")
    LOG.info("   Reprojection error: {:.3f} pixels".format(rep_err))
    LOG.info("   Camera matrix:\n{}".format(cmtx))
    LOG.info("   Distortion coeffs:\n{}".format(distk))
    return img_shape, cmtx, distk


# /opt/ros/melodic/lib/camera_calibration/tarfile_calibration.py --mono --visualize -q 0.025 -s 8x6 /tmp/enac_drone_july_4_2019.tgz
def write_calib(img_shape, cmtx, distk, cam_calib_path, a=1.):
    R = np.eye(3, dtype=np.float64)
    P = np.zeros((3, 4), dtype=np.float64)
    ncm, _ = cv2.getOptimalNewCameraMatrix(cmtx, distk, img_shape, a)
    for j in range(3):
        for i in range(3):
            P[j,i] = ncm[j, i]
    print(P)
    pdb.set_trace()

    cam_info_msg = sensor_msgs.msg.CameraInfo()
    cam_info_msg.width = img_shape[0]
    cam_info_msg.height = img_shape[1]
    cam_info_msg.K = np.ravel(cmtx).copy().tolist()
    cam_info_msg.D = np.ravel(distk).copy().tolist()
    cam_info_msg.R = np.ravel(R).copy().tolist()
    cam_info_msg.P = np.ravel(P).copy().tolist()
    smocap.camera.write_extrinsics(cam_calib_path, cam_info_msg, 'ueye_drone')
    
def calibrate_extrinsic(intr_cam_calib_path, extr_img_path, extr_pts_path):
    
    img = cv2.imread(extr_img_path,  cv2.IMREAD_UNCHANGED)
    camera_matrix, dist_coeffs, w, h = smocap.camera.load_intrinsics(intr_cam_calib_path)
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    cv2.imwrite('/tmp/foo.png', img_undistorted)
    
    pts_name, pts_img, pts_world = trrvu.read_point(extr_pts_path)
    #pdb.set_trace()
    pts_img_undistorted = cv2.undistortPoints(pts_img.reshape((-1, 1, 2)), camera_matrix, dist_coeffs, None, new_camera_matrix).squeeze()
            
    (success, rotation_vector, translation_vector) = cv2.solvePnP(pts_world, pts_img.reshape(-1, 1, 2),
                                                                  camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    LOG.info("PnP {} {} {}".format(success, rotation_vector.squeeze(), translation_vector.squeeze()))

    rep_pts_img =  cv2.projectPoints(pts_world, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0].squeeze()
    rep_err = np.mean(np.linalg.norm(pts_img - rep_pts_img, axis=1))
    LOG.info('reprojection error {} px'.format(rep_err))

    world_to_camo_T = smocap.utils.T_of_t_r(translation_vector.squeeze(), rotation_vector)
    world_to_camo_t, world_to_camo_q = smocap.utils.tq_of_T(world_to_camo_T)
    print(' world_to_camo_t {} world_to_camo_q {}'.format(world_to_camo_t, world_to_camo_q))

    camo_to_world_T = np.linalg.inv(world_to_camo_T)
    camo_to_world_t, camo_to_world_q = smocap.utils.tq_of_T(camo_to_world_T)
    print(' cam_to_world_t {} cam_to_world_q {}'.format(camo_to_world_t, camo_to_world_q))

    T_c2co = np.array([[  0.,  -1.,  0. , 0],
                       [  0.,   0., -1. , 0],
                       [  1.,   0.,  0. , 0],
                       [  0.,   0.,  0. , 1.]])
    T_co2c =  np.linalg.inv(T_c2co)
    caml_to_ref_T = np.dot(camo_to_world_T, T_c2co)
    caml_to_ref_t, caml_to_ref_q = smocap.utils.tq_of_T(caml_to_ref_T)
    print(' caml_to_ref_t {} caml_to_ref_q {}'.format(caml_to_ref_t, caml_to_ref_q))
    caml_to_ref_rpy = np.asarray(tf.transformations.euler_from_matrix(caml_to_ref_T, 'sxyz'))
    print(' caml_to_ref_rpy {}'.format(caml_to_ref_rpy))

    apertureWidth, apertureHeight = 6.784, 5.427
    fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(camera_matrix, (h, w), apertureWidth, apertureHeight) 
    print(fovx, fovy, focalLength, principalPoint, aspectRatio)
    
    def draw(cam_img, pts_id, keypoints_img, rep_keypoints_img, pts_world):
        for i, p in enumerate(keypoints_img.astype(int)):
            cv2.circle(cam_img, tuple(p), 1, (0,255,0), -1)
            cv2.putText(cam_img, '{}'.format(pts_id[i][:1]), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(cam_img, '{}'.format(pts_world[i][:2]), tuple(p+[0, 25]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        for i, p in enumerate(rep_keypoints_img.astype(int)):
            cv2.circle(cam_img, tuple(p), 1, (0,0,255), -1)

    draw(img, pts_name, pts_img, rep_pts_img, pts_world)
    cv2.imshow('original image', img)

    for i, p in enumerate(pts_img_undistorted.astype(int)):
        cv2.circle(img_undistorted, tuple(p), 1, (0,255,0), -1)
    
    cv2.imshow('undistorted image', img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #intr_cam_calib_path = '/home/poine/.ros/camera_info/ueye_drone.yaml'
    #extr_img_path = /home/poine/work/robot_data/jeanmarie/floor_tiles_03.png
    #extr_pts_path = '/home/poine/work/roverboard/roverboard_guidance/config/be_calib_data_tiles_03.yaml'
    intr_cam_calib_path = '/home/poine/.ros/camera_info/camera_road_front.yaml'
    extr_img_path = '/home/poine/work/robot_data/caroline/floor_tiles_z_01.png'
    extr_pts_path = '/home/poine/work/roverboard/roverboard_caroline/config/ext_calib_pts_floor_tiles_01.yaml'
    if 0:
        img_shape, cmtx, distk = calibrate_intrinsic()
        write_calib(img_shape, cmtx, distk, intr_cam_calib_path)
    if 1:
        calibrate_extrinsic(intr_cam_calib_path, extr_img_path, extr_pts_path)
