#!/usr/bin/env python
import os, time, math, numpy as np

import cv2
from matplotlib import pyplot as plt
import yaml
import shapely, shapely.geometry

import two_d_guidance.trr.utils as trr_u, smocap
import pdb

# fit an image to a new canvas (changes the size)
def change_canvas(img_in, out_h, out_w, border_color=128):
    img_out = np.full((out_h, out_w, 3), border_color, dtype=np.uint8)
    in_h, in_w, _ = img_in.shape
    scale = min(float(out_h)/in_h, float(out_w)/in_w)
    h, w = int(scale*in_h), int(scale*in_w)
    dx, dy = (out_w-w)/2, (out_h-h)/2
    img_out[dy:dy+h, dx:dx+w] = cv2.resize(img_in, (w, h))
    return img_out

# loads a camera - this calls smocap stuff - should probably go there too
def load_cam_from_files(intr_path, extr_path, cam_name='cam1', alpha=1.):
    cam = smocap.Camera(0, 'cam1')
    cam.load_intrinsics(intr_path)
    cam.set_undistortion_param(alpha)
    cam.load_extrinsics(extr_path)
    return cam

# read an array of points - this is used for extrinsic calibration
def read_point(yaml_data_path):
    with open(yaml_data_path, 'r') as stream:
        ref_data = yaml.load(stream)
    pts_name, pts_img, pts_world = [], [], []
    for _name, _coords in ref_data.items(): 
        pts_img.append(_coords['img'])
        pts_world.append(_coords['world'])
        pts_name.append(_name)
    return pts_name, np.array(pts_img, dtype=np.float64), np.array(pts_world, dtype=np.float64)

# Computes the bridge filtering. This code is speed optimized. Original code using filter2D is much slower (but much easier to understand):
# KEEP        cell = np.ones((cell_height, cell_width), np.uint8)
# KEEP        kernel = np.concatenate((cell * -0.5, cell, cell * -0.5), axis = 1) / (cell_height * cell_width)
# KEEP        return cv2.filter2D(mini_img, -1, kernel)
def bridge_filter(img, cell_width, cell_height):
    smooth = cv2.boxFilter(img, cv2.CV_16S, (cell_width, cell_height))
    half = smooth / 2
    smooth[:, cell_width:-cell_width] -= half[:, 0:-2*cell_width] + half[:, 2*cell_width:]
    # This code avoids to build half and divide by 2, but implies complex rewrite for the last part: cv2.scaleAdd(smooth[:, 0:-2*cell_width] + smooth[:, 2*cell_width:], -0.5, smooth[:, cell_width:-cell_width], smooth[:, cell_width:-cell_width])
    smooth[:, 0:cell_width] -= half[:, cell_width:2*cell_width]
    smooth[:, -cell_width:] -= half[:, -2*cell_width:-cell_width]
    for c in range(cell_width):
        smooth[:, c] -= half[:, 0]
        smooth[:, -c-1] -= half[:, -1]
    smooth[smooth < 0] = 0
    return np.uint8(smooth)



class BirdEyeParam:
    def __init__(self, x0=0.3, dx=3., dy=2., w=640):
        # coordinates of viewing area on local floorplane in base_footprint frame
        self.x0, self.dx, self.dy = x0, dx, dy
        # coordinates of viewing area as a pixel array (unwarped)
        self.w = w; self.s = self.dy/self.w; self.h = int(self.dx/self.s)

        # viewing area in base_footprint frame
        # bottom_right, top_right, top_left, bottom_left in base_footprint frame
        self.corners_be_blf = np.array([(self.x0, self.dy/2, 0.), (self.x0+self.dx, self.dy/2, 0.), (self.x0+self.dx, -self.dy/2, 0.), (self.x0, -self.dy/2, 0.)])
        self.corners_be_img = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])
        
class CarolineBirdEyeParam(BirdEyeParam):
    #def __init__(self, x0=0.4, dx=0.9, dy=0.2, w=480): # inside
    #def __init__(self, x0=0.28, dx=0.9, dy=0.8, w=480): # real
    def __init__(self, x0=0.1, dx=2., dy=1.5, w=480):  # outside
        BirdEyeParam.__init__(self, x0, dx, dy, w)

class ChristineBirdEyeParam(BirdEyeParam):
    #def __init__(self, x0=0.30, dx=4., dy=2., w=480):
    def __init__(self, x0=0.30, dx=3., dy=2., w=640):
        BirdEyeParam.__init__(self, x0, dx, dy, w)
        
def NamedBirdEyeParam(_name):
    if    _name == 'caroline':  return CarolineBirdEyeParam()
    elif  _name == 'christine': return ChristineBirdEyeParam()
    return None

def _make_line(p0, p1, spacing=1, endpoint=True):
    dist = np.linalg.norm(p1-p0)
    n_pt = dist/spacing
    if endpoint: n_pt += 1
    return np.stack([np.linspace(p0[j], p1[j], n_pt, endpoint=endpoint) for j in range(len(p0))], axis=-1)

def _lines_of_corners(corners, spacing):
    return np.concatenate([_make_line(corners[i-1], corners[i], spacing=spacing, endpoint=False) for i in range(len(corners))])

class BirdEyeTransformer:
    def __init__(self, cam, be_param):
        self.set_param(cam, be_param)
        self.cnt_fp = None
        self.unwarped_img = None
        
    def set_param(self, cam, be_param):
        print('setting params x0 {} dx {} dy {}'.format(be_param.x0, be_param.dx, be_param.dy))
        self.param = be_param
        self.w, self.h = be_param.w, be_param.h
        self._compute_cam_viewing_area(cam)
        self._compute_H(cam)
        self._compute_H2(cam)

    def _compute_cam_viewing_area(self, cam, max_dist=6):
        # compute the contour of the intersection between camera frustum and floor plane
        corners_cam_img = np.array([[0., 0], [cam.w, 0], [cam.w, cam.h], [0, cam.h], [0, 0]])
        borders_cam_img = _lines_of_corners(corners_cam_img, spacing=1)
        borders_undistorted = cam.undistort_points(borders_cam_img.reshape(-1, 1, 2))
        borders_cam = np.array([np.dot(cam.inv_undist_K, [u, v, 1]) for (u, v) in borders_undistorted.squeeze()])
        borders_floor_plane_cam = get_points_on_plane(borders_cam, cam.fp_n, cam.fp_d)
        in_frustum_idx = np.logical_and(borders_floor_plane_cam[:,2]>0, borders_floor_plane_cam[:,2]<max_dist)
        borders_floor_plane_cam = borders_floor_plane_cam[in_frustum_idx,:]
        self.borders_floor_plane_world = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in borders_floor_plane_cam])

        # compute intersection between camera view and bird eye areas
        poly_va_blf = shapely.geometry.Polygon(self.borders_floor_plane_world[:,:2])
        poly_be_blf = shapely.geometry.Polygon(self.param.corners_be_blf[:,:2])
        foo = poly_va_blf.intersection(poly_be_blf).exterior.coords.xy
        self.borders_isect_be_cam = np.zeros((len(foo[0]), 3))
        self.borders_isect_be_cam[:,:2] = np.array(foo).T

        # compute masks
        if 0:  # TODO
            self.mask_blf = np.array([(0, 0), (0.3, -0.3), (0, -0.3), (0, 0)])
            #pdb.set_trace()
            self.mask_unwraped = self.lfp_to_unwarped2(cam, self.mask_blf).reshape((1,-1,2)).astype(np.int64)
            print self.mask_unwraped
        else:
            y0, y1, y2, y3 = 700, 940, 960, 650
            x1, x2, x3 = 260, 360, 640
            self.mask_unwraped = np.array([[(0, y0), (x1, y1), (x2, y1), (x3, y3), (x3, y2), (0, y2)]])

    def _img_point_in_camera_frustum(self, pts_img, cam): return np.logical_and(pts_img[:,0,0] > 0, pts_img[:,0,0] < cam.w)

    # compute Homography from image plan to be image (unwarped)
    def _compute_H(self, cam, precomp_path='/tmp/be_precomp'):
        try:  # speedup startup by loading precomputed values if they exists
            data =  np.load(precomp_path+'.npz')
            print('found precomputed data in {}'.format(precomp_path))
            self.H = data['H']
            self.unwrapped_xmap = data['unwrapped_xmap']
            self.unwrapped_ymap = data['unwrapped_ymap']
            self.unwrap_undist_xmap_int = data['unwrap_undist_xmap']
            self.unwrap_undist_ymap_int = data['unwrap_undist_ymap']
        except IOError:  
            print('precomputed data {} does not exist'.format(precomp_path))
            va_corners_img  = cam.project(self.borders_isect_be_cam)
            va_corners_imp  = cam.undistort_points(va_corners_img)
            va_corners_unwarped = self.lfp_to_unwarped(cam, self.borders_isect_be_cam.squeeze())
            self.H, status = cv2.findHomography(srcPoints=va_corners_imp, dstPoints=va_corners_unwarped, method=cv2.RANSAC, ransacReprojThreshold=0.01)
            print('computed H unwarped: ({}/{} inliers)\n{}'.format( np.count_nonzero(status), len(va_corners_imp), self.H))
            print('computing H maps')
            # make homography maps
            self.unwrapped_xmap = np.zeros((self.h, self.w), np.float32)
            self.unwrapped_ymap = np.zeros((self.h, self.w), np.float32)
            Hinv = np.linalg.inv(self.H)
            for y in range(self.h):
                for x in range(self.w):
                    pt_be = np.array([[x], [y], [1]], dtype=np.float32)
                    pt_imp = np.dot(Hinv, pt_be)
                    pt_imp /= pt_imp[2]
                    self.unwrapped_xmap[y,x], self.unwrapped_ymap[y,x] =  pt_imp[:2]

            # make combined undistortion + homography maps
            self.unwrap_undist_xmap = np.zeros((self.h, self.w), np.float32)
            self.unwrap_undist_ymap = np.zeros((self.h, self.w), np.float32)
            for y in range(self.h):
                for x in range(self.w):
                    pt_be = np.array([[x, y, 1]], dtype=np.float32).T
                    pt_imp = np.dot(cam.inv_undist_K, np.dot(Hinv, pt_be))
                    pt_imp /= pt_imp[2]
                    pt_img = cv2.projectPoints(pt_imp.T, np.zeros(3), np.zeros(3), cam.K, cam.D)[0]
                    self.unwrap_undist_xmap[y,x], self.unwrap_undist_ymap[y,x] = pt_img.squeeze()
            self.unwrap_undist_xmap_int, self.unwrap_undist_ymap_int = cv2.convertMaps(self.unwrap_undist_xmap, self.unwrap_undist_ymap, cv2.CV_16SC2)
            print('dome computing H maps, saving for reuse')
            np.savez(precomp_path, H=self.H, unwrapped_xmap=self.unwrapped_xmap, unwrapped_ymap=self.unwrapped_ymap,
                     unwrap_undist_xmap=self.unwrap_undist_xmap_int,  unwrap_undist_ymap=self.unwrap_undist_ymap_int)

    # compute Homography from image plan to base link footprint
    def _compute_H2(self, cam):
        print('### bird eye compute H for be_blf\narea:\n{}'.format(self.param.corners_be_blf))
        va_corners_img  = cam.project(self.borders_isect_be_cam)
        va_corners_imp  = cam.undistort_points(va_corners_img)
        pts_src = va_corners_imp.squeeze()
        pts_dst = self.borders_isect_be_cam.squeeze()
        self.H2, status = cv2.findHomography(srcPoints=va_corners_imp, dstPoints=self.borders_isect_be_cam, method=cv2.RANSAC, ransacReprojThreshold=0.01)
        print('computed H blf: ({}/{} inliers)\n{}'.format(np.count_nonzero(status), len(va_corners_imp), self.H2))
             
        
    def plot_calibration(self, img_undistorted):
        pass
        
    def points_imp_to_be(self, points_imp):
        return cv2.perspectiveTransform(points_imp, self.H)

    def unwarped_to_fp(self, cam, cnt_uw):
        s = self.param.dy/self.param.w
        self.cnt_fp = np.array([((self.param.h-p[1])*s+self.param.x0, (self.param.w/2-p[0])*s, 0.) for p in cnt_uw.squeeze()])
        return self.cnt_fp

    def lfp_to_unwarped(self, cam, cnt_lfp):
        s = self.param.dy/self.param.w
        cnt_uv = np.array([(self.param.w/2-y/s, self.param.h-(x-self.param.x0)/s) for x, y, _ in cnt_lfp])
        return cnt_uv

    def lfp_to_unwarped2(self, cam, cnt_lfp):
        s = self.param.dy/self.param.w
        cnt_uv = np.array([(self.param.w/2-y/s, self.param.h-(x-self.param.x0)/s) for x, y in cnt_lfp])
        return cnt_uv
    

    def points_imp_to_blf(self, points_imp):
        return cv2.perspectiveTransform(points_imp, self.H2)
    

    def process(self, img):
        #pdb.set_trace()
        self.unwarped_img = cv2.warpPerspective(img, self.H, (self.w, self.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return self.unwarped_img

    def unwarp_map(self, img):
        self.unwarped_img = cv2.remap(img, self.unwrapped_xmap, self.unwrapped_ymap, cv2.INTER_LINEAR)
        return self.unwarped_img

    def undist_unwarp_map(self, img, cam): # FIXME do that with a single map
        #undistorted_img = cam.undistort_img_map(img)
        #self.unwarped_img = cv2.remap(undistorted_img, self.unwrapped_xmap, self.unwrapped_ymap, cv2.INTER_LINEAR)
        #self.unwarped_img = cv2.remap(img, self.unwrap_undist_xmap, self.unwrap_undist_ymap, cv2.INTER_LINEAR)
        self.unwarped_img = cv2.remap(img, self.unwrap_undist_xmap_int, self.unwrap_undist_ymap_int, cv2.INTER_LINEAR)
        return self.unwarped_img
    
    def draw_debug(self, cam, img=None, lane_model=None, cnts_be=None, border_color=128, fill_color=(255, 0, 255)):
        if img is None:
            img = np.zeros((self.h, self.w, 3), dtype=np.float32) # black image in be coordinates
        # elif img.dtype == np.uint8:
        #     img = img.astype(np.float32)/255.
        if  cnts_be is not None:
            for cnt_be in cnts_be:
                cnt_be_int = cnt_be.astype(np.int32)
                cv2.drawContours(img, cnt_be_int, -1, (255,0,255), 3)
                cv2.fillPoly(img, pts =[cnt_be_int], color=fill_color)
        if lane_model is not None and lane_model.is_valid():
            self.draw_lane(cam, img, lane_model, lane_model.x_min, lane_model.x_max)
        return change_canvas(img, cam.h, cam.w)

    def draw_lane(self, cam, img, lane_model, l0=None, l1=None, color=(0,128,0)):
        if l0 is None:
            l0, l1 = lane_model.x_min, lane_model.x_max 
        # Coordinates in local_floor_plane(aka base_link_footprint) frame
        xs = np.linspace(l0, l1, 20); ys = lane_model.get_y(xs)
        pts_lfp = np.array([[x, y, 0] for x, y in zip(xs, ys)])
        pts_img = self.lfp_to_unwarped(cam, pts_lfp)
        #pdb.set_trace()
        for i in range(len(pts_img)-1):
            try:
                #print(tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)))
                cv2.line(img, tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)), color, 4)
            except OverflowError:
                pass





class ColoredBlobDetector:
    param_names = [
        'blobColor',
        'filterByArea',
        'filterByCircularity',
        'filterByColor',
        'filterByConvexity',
        'filterByInertia',
        'maxArea',
        'maxCircularity',
        'maxConvexity',
        'maxInertiaRatio',
        'maxThreshold',
        'minArea',
        'minCircularity',
        'minConvexity',
        'minDistBetweenBlobs',
        'minInertiaRatio',
        'minRepeatability',
        'minThreshold',
        'thresholdStep' ]
    
    def __init__(self, hsv_range, cfg_path=None):
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.set_hsv_range(hsv_range)
        if cfg_path is not None:
            self.load_cfg(cfg_path)
        else:
            self.detector = cv2.SimpleBlobDetector_create(self.blob_params)
        self.keypoints = []

    def set_hsv_range(self, hsv_range):
        self.hsv_range = hsv_range
        
    def load_cfg(self, path):
        print('loading config {}'.format(path))
        with open(path, 'r') as stream:   
            d = yaml.load(stream)
        for k in d:
            setattr(self.blob_params, k, d[k])
        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)

    def save_cfg(self, path):
        d = {}
        for p in ColoredBlobDetector.param_names:
            d[p] =  getattr(self.params, p)
        with open(path, 'w') as stream:
            yaml.dump(d, stream, default_flow_style=False)

    def update_param(self, name, value):
        setattr(self.blob_params, name, value)
        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)
            
    def process_hsv_image(self, hsv_img):
        masks = [cv2.inRange(hsv_img, hsv_min, hsv_max) for (hsv_min, hsv_max) in self.hsv_range]
        self.mask = np.sum(masks, axis=0).astype(np.uint8)
        self.keypoints = self.detector.detect(self.mask)
        #print('detected {}'.format(self.keypoints))
        return self.keypoints, None

    def keypoints_nb(self): return len(self.keypoints)
    
    def draw(self, img, color):
        if len(self.keypoints) > 0:
            img_with_keypoints = cv2.drawKeypoints(img, self.keypoints, outImage=np.array([]), color=color,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            for k in self.keypoints:
                cv2.circle(img, tuple(np.array(k.pt).astype(np.int)), radius=int(k.size), color=color, thickness=2, lineType=8, shift=0)
                #if color == (0, 0, 255): print k.pt, k.size
            
                #print([kp.pt for kp in keypoints])
                return img_with_keypoints
            else: return img

class ContourFinder:
    def __init__(self, min_area=None, single_contour=False):
        self.min_area = min_area
        self.single_contour = single_contour
        self.img = None
        self.cnts = None
        self.cnt_max = None
        self.cnt_max_area = 0
        self.valid_cnts = None

    def has_contour(self): return (self.cnt_max is not None)
    def get_contour(self): return self.cnt_max
    def get_contour_area(self): return self.cnt_max_area
    
    def process(self, img):
        # detect contours
        self.img2, self.cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        # TODO: remove cnt_max computing
        # TODO: remove single_contour code
        self.cnt_max = None
        if self.cnts is not None and len(self.cnts) > 0:
            # This reduce global computing time but may impact polyfit
            #self.cnts = [cv2.approxPolyDP(c, 0.8, True) for c in self.cnts]
            # find contour with largest area
            self.cnt_max = max(self.cnts, key=cv2.contourArea)
            self.cnt_max_area = cv2.contourArea(self.cnt_max)
            if self.min_area is not None and self.cnt_max_area < self.min_area:
                self.cnt_max, self.cnt_max_area = None, 0

            # find all contours with a sufficient area
            if not self.single_contour:
                self.cnt_areas = np.array([cv2.contourArea(c) for c in self.cnts])
                self.valid_cnts_idx = self.cnt_areas > self.min_area
                self.valid_cnts = np.array(self.cnts)[self.valid_cnts_idx]
               
    def draw(self, img, mc_border_color=(255,0,0), thickness=3, fill=True, mc_fill_color=(255,0,0), draw_all=False,
             ac_fill_color=(0, 128, 128)):
        if self.cnt_max is not None:
            if fill:
                try:
                    cv2.fillPoly(img, pts =[self.cnt_max], color=mc_fill_color)
                except cv2.error: # fails when poly is too small?
                    #print self.cnt_max.shape
                    pass
            cv2.drawContours(img, self.cnt_max, -1, mc_border_color, thickness)

    def draw2(self, img, cnts, cnts_mask, fill_color1=(0, 128, 0), fill_color2=(0, 0, 128)):
        for c, inlier in zip(cnts, cnts_mask):
            cv2.drawContours(img, c, -1, (255, 0, 0), 2)
            cv2.fillPoly(img, pts =[c], color=fill_color1 if inlier else fill_color2)
            

class ColoredContourDetector:
    def __init__(self, hsv_range, min_area=None, gray_tresh=150):
        #self.mask_extractor = ColoredMaskExtractor(hsv_range)
        self.mask_extractor = ColoredAndLightMaskExtractor(hsv_range)
        self.bin_ctr_finder = ContourFinder(min_area)

    def set_hsv_range(self, hsv_range):
        self.mask_extractor.set_hsv_range(hsv_range)
    def set_gray_threshold(self, v): 
        self.mask_extractor.set_threshold(v)
        
    def has_contour(self): return self.bin_ctr_finder.has_contour()
    def get_contour(self): return self.bin_ctr_finder.get_contour()
    def get_contour_area(self): return self.bin_ctr_finder.get_contour_area()
        
    #def process_bgr_image(self, bgr_img): return self.process_hsv_image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV))
    #def process_rgb_image(self, rgb_img): return self.process_hsv_image(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV))
    def process_hsv_image(self, hsv_img, img_gray):
        self.mask = self.mask_extractor.process_hsv_image(hsv_img, img_gray)
        #self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        #self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
        self.bin_ctr_finder.process(self.mask_extractor.mask)
            
            
    def draw(self, bgr_img, color=(255,0,0), fill_color=(255,255,0)):
        self.bin_ctr_finder.draw(bgr_img, thickness=1, mc_border_color=color, mc_fill_color=fill_color)
        return bgr_img

class ColoredMaskExtractor:
    def __init__(self,  hsv_range):
        self.set_hsv_range(hsv_range)
        self.mask = None
        
    def set_hsv_range(self, hsv_range):
        self.hsv_range = hsv_range

    def process_hsv_image(self, hsv_img, img_gray):
        masks = [cv2.inRange(hsv_img, hsv_min, hsv_max) for (hsv_min, hsv_max) in self.hsv_range]
        self.mask = np.sum(masks, axis=0).astype(np.uint8)
        return self.mask

class ColoredAndLightMaskExtractor(ColoredMaskExtractor):
    def __init__(self,  hsv_range, gray_thresh=150):
        ColoredMaskExtractor.__init__(self,  hsv_range)
        self.grey_mask = BinaryThresholder(thresh=gray_thresh)

    def set_threshold(self, v): self.grey_mask.set_threshold(v)

    def process_hsv_image(self, hsv_img, img_gray):
        ColoredMaskExtractor.process_hsv_image(self, hsv_img, img_gray)
        self.mask += self.grey_mask.process(img_gray)
        return self.mask
        
#
# HSV range for different colors in cv conventions (h in [0 180], s and v in [0 255]
#
# Red is 0-30 and  150-180 values.
# we limit to 0-10 and 170-180
def hsv_red_range(hc=0, hsens=10, smin=100, smax=255, vmin=30, vmax=255):
    # red_range = [[np.array([0,smin,vmin]), np.array([hsens,smax,vmax])],
    #               [np.array([180-hsens,smin,vmin]), np.array([180,smax,vmax])]]
    red_range = hsv_range(hc, hsens, smin, smax, vmin, vmax)
    return red_range
# Yellow is centered on h=30
def hsv_yellow_range(hsens=10, smin=100, smax=255, vmin=100, vmax=255):
    return hsv_range(30, hsens, smin, smax, vmin, vmax)
# Green is centered on h=60
def hsv_green_range(hsens=10, smin=100, smax=255, vmin=100, vmax=255):
    return hsv_range(60, hsens, smin, smax, vmin, vmax)
# Blue is centered on h=110
def hsv_blue_range(hsens=10, smin=50, smax=255, vmin=50, vmax=255):
    return hsv_range(110, hsens, smin, smax, vmin, vmax)

def hsv_range(hcenter, hsens, smin, smax, vmin, vmax):
    # compute hsv range, wrap around 180 if needed (returns bottom and top range in this case)
    hmin, hmax = hcenter-hsens, hcenter+hsens
    ranges = []
    if hmin < 0:
        ranges.append([np.array([0, smin, vmin]), np.array([hmax, smax, vmax])])
        ranges.append([np.array([hmin+180, smin, vmin]), np.array([180, smax, vmax])])
    elif hmax > 180:
        ranges.append([np.array([0, smin, vmin]),    np.array([hmax-180, smax, vmax])])
        ranges.append([np.array([hmin, smin, vmin]), np.array([180, smax, vmax])])
    else:
        ranges.append([np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax])])
    return ranges
    

class StartFinishDetector:
    CTR_START, CTR_FINISH, CTR_NB = range(3)
    def __init__(self):
        self.green_ccf = ColoredContourDetector(hsv_green_range(), min_area=100)
        self.red_ccf = ColoredContourDetector(hsv_red_range(), min_area=100)
        self.ccfs = [self.green_ccf, self.red_ccf]

    def sees_ctr(self, ctr_idx): return self.ccfs[ctr_idx].has_contour()
    def sees_start(self): return self.sees_ctr(self.CTR_START)
    def sees_finish(self): return self.sees_ctr(self.CTR_FINISH)
    def get_contour(self, ctr_idx): return self.ccfs[ctr_idx].get_contour()
        
    def process_image(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        for ccf in self.ccfs:
            ccf.process_hsv_image(hsv, gray)
 
    def draw(self, bgr_img):
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        bgr_img2 = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        fill_colors = [(0, 255,0), (0,0,255)]
        for ccf, fill_color in zip(self.ccfs, fill_colors):
            ccf.draw(bgr_img2, color=(155, 155, 0), fill_color=fill_color)
        return bgr_img2


class BinaryThresholder:
    def __init__(self, thresh=200):
        self.thresh_val = thresh
        self.threshold = None
        
    def process(self, img):
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        #blurred = cv2.GaussianBlur(img, (25, 25), 0)
        #self.threshold = blurred
        ret, self.threshold = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)
        #ret, self.threshold = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #self.threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -30)
        #self.threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -30)
        return self.threshold

    def process_bgr(self, img, birdeye_mode=True):
        blue_img = img[:, :, 0]
        if birdeye_mode:
            width = 20
            bridge_img = bridge_filter(blue_img, width, width)
            # Level -10 for inside, -15 for outside
            self.threshold = cv2.adaptiveThreshold(bridge_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 351, -15)
        else:
            res = []
            # Calibration for christine
            for step in ((0, 35, 8), (35, 70, 13), (70, 120, 22), (120, 200, 35), (200, 300, 50), (300, None, 70)):
                width = step[2]
                h = min(width, 50)
                mini_img = blue_img[step[0]:step[1], :]
                bridge_img = bridge_filter(mini_img, width, h)
                # Level -10 for inside, -15 for outside
                bridge_th_img = cv2.adaptiveThreshold(bridge_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 351, -15)
                res.append(bridge_th_img)
            self.threshold = cv2.vconcat(res)
            
        return self.threshold    
            
    def set_threshold(self, thresh): self.thresh_val = thresh

            
class BlackWhiteThresholder:
    def __init__(self):
        self.low, self.high = 196, 255
        self.threshold = None
        
    def process(self, img):
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        light_white, dark_white = (0, 0, 180), (25, 36, 255) #hsv
        self.mask_white = cv2.inRange(hsv_img, light_white, dark_white)
        #h, s, v = cv2.split(hsv_nemo)
        #ret, self.threshold = cv2.threshold(blurred, self.low, self.high, 0)
        return img#self.threshold

        
class HoughLinesFinder:
    def __init__(self, mask):
        self.img, self.lines, self.vlines = None, None, None
        y0, y1, y2 = 850, 950, 1000
        x1, x2, x3 = 110, 350, 480 
        self.mask = mask#np.array([[(0, y0), (x1, y1), (x2, y1), (x3, y0), (x3, y2), (0, y2)]])
        self.use_mask=True
        
    def process(self, img, minval=100, maxval=200):
        self.edges = cv2.Canny(img, minval, maxval, apertureSize=3, L2gradient=False)
        if self.use_mask: cv2.fillPoly(self.edges, pts =self.mask, color=(0))
        rho_acc, theta_acc = 6, np.deg2rad(1)
        lines, threshold = 80, 30
        minLineLength, maxLineGap = 50, 25
        self.lines = cv2.HoughLinesP(self.edges, rho=rho_acc, theta=theta_acc , threshold=threshold,
                                     lines=lines, minLineLength=minLineLength, maxLineGap=maxLineGap)
        if self.lines is None or len(self.lines) == 0:
            self.vlines = np.array([]); return
        
        print('found {} lines'.format(len(self.lines)))
        self.vlines = []
        dth=45.
        for line in self.lines:
            x1,y1,x2,y2 = line.squeeze()
            angle = np.arctan2(y2 - y1, x2 - x1)
            if angle < np.deg2rad(90+dth) and angle > np.deg2rad(90-dth) or\
               angle > -np.deg2rad(90+dth) and angle < -np.deg2rad(dth-30) :
                self.vlines.append(line)
        self.vlines = np.array(self.vlines)
        print('kept {} vlines'.format(len(self.vlines)))
        #pdb.set_trace()

    def has_lines(self): return self.vlines is not None and len(self.vlines) > 0
        
    def draw(self, img, color=(0,0,255), draw_mask=True, draw_lines=False):
        cv2.polylines(img, self.mask, isClosed=True, color=(0, 255, 255), thickness=2)
        if self.lines is None: return
        for line in self.lines:
            x1,y1,x2,y2 = line.squeeze()
            cv2.line(img,(x1,y1),(x2,y2),color,2)
        if draw_lines:
            for line in self.vlines:
                x1,y1,x2,y2 = line.squeeze()
                cv2.line(img,(x1,y1),(x2,y2),(255, 0, 0),2)

            
def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])

class FloorPlaneInjector:
    def __init__(self):
        self.contour_floor_plane_blf = None

    def compute(self, contour_img, cam):
        # undistorted coordinates
        #contour_undistorted = cv2.undistortPoints(contour_img.astype(np.float32), cam.K, cam.D, None, cam.undist_K)
        contour_undistorted = cam.undistort_points(contour_img.astype(np.float32))
        # contour in optical plan
        contour_camo = [np.dot(cam.inv_undist_K, [u, v, 1]) for (u, v) in contour_undistorted.squeeze()]
        # contour projected on floor plane (in camo frame)
        contour_floor_plane_camo = get_points_on_plane(contour_camo, cam.fp_n, cam.fp_d)
        # contour projected on floor plane (in body frame)
        self.contour_floor_plane_blf = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in contour_floor_plane_camo])

        return self.contour_floor_plane_blf

        
# Timing of pipelines
class Pipeline:
    def __init__(self):
        self.skipped_frames = 0
        self.last_seq = None
        self.last_stamp = None
        self.cur_fps = 0.
        self.min_fps, self.max_fps, self.lp_fps = np.inf, 0, 0.1
        self.last_processing_duration = None
        self.min_proc, self.max_proc, self.lp_proc = np.inf, 0, 1e-6
        self.idle_t = 0.
        self.k_lp = 0.9 # low pass coefficient
        
    def process_image(self, img, cam, stamp, seq):
        if self.last_stamp is not None:
            _dt = (stamp - self.last_stamp).to_sec()
            if np.abs(_dt) > 1e-9:
                self.cur_fps = 1./_dt
                self.min_fps = np.min([self.min_fps, self.cur_fps])
                self.max_fps = np.max([self.max_fps, self.cur_fps])
                self.lp_fps  = self.k_lp*self.lp_fps+(1-self.k_lp)*self.cur_fps
        self.last_stamp = stamp
        if self.last_seq is not None:
            self.skipped_frames += seq-self.last_seq-1
        self.last_seq = seq

        _start = time.time()
        self._process_image(img, cam)
        _end = time.time()

        self.last_processing_duration = _end-_start
        self.min_proc = np.min([self.min_proc, self.last_processing_duration])
        self.max_proc = np.max([self.max_proc, self.last_processing_duration])
        self.lp_proc = self.k_lp*self.lp_proc+(1-self.k_lp)*self.last_processing_duration
        self.idle_t = 1./self.lp_fps - self.lp_proc

    def draw_timing(self, img, x0=280, y0=20, dy=35, h=0.75, color_bgr=(220, 220, 50)):
        f, c, w = cv2.FONT_HERSHEY_SIMPLEX, color_bgr, 2
        try: 
            txt = 'fps: {:.1f} (min {:.1f} max {:.1f})'.format(self.lp_fps, self.min_fps, self.max_fps)
            cv2.putText(img, txt, (x0, y0), f, h, c, w)
            txt = 'skipped: {:d} (cpu {:.3f}/{:.3f}s)'.format(self.skipped_frames, self.lp_proc, 1./self.lp_fps)
            cv2.putText(img, txt, (x0, y0+dy), f, h, c, w)
        except AttributeError: pass
        






 
