#!/usr/bin/env python
import time, math, numpy as np

import cv2
from matplotlib import pyplot as plt
import yaml

import two_d_guidance.trr.utils as trr_u, smocap
import pdb

def change_canvas(img_in, out_h, out_w, border_color=128):
    img_out = np.full((out_h, out_w, 3), border_color, dtype=np.uint8)
    in_h, in_w, _ = img_in.shape
    scale = min(float(out_h)/in_h, float(out_w)/in_w)
    h, w = int(scale*in_h), int(scale*in_w)
    dx, dy = (out_w-w)/2, (out_h-h)/2
    img_out[dy:dy+h, dx:dx+w] = cv2.resize(img_in, (w, h))
    return img_out

def load_cam_from_files(intr_path, extr_path, cam_name='cam1', alpha=1.):
    cam = smocap.Camera(0, 'cam1')
    cam.load_intrinsics(intr_path)
    cam.set_undistortion_param(alpha)
    cam.load_extrinsics(extr_path)
    return cam

# read an array of points - this is used for bireye calibration
def read_point(yaml_data_path):
    with open(yaml_data_path, 'r') as stream:
        ref_data = yaml.load(stream)
    pts_name, pts_img, pts_world = [], [], []
    for _name, _coords in ref_data.items(): 
        pts_img.append(_coords['img'])
        pts_world.append(_coords['world'])
        pts_name.append(_name)
    return pts_name, np.array(pts_img, dtype=np.float64), np.array(pts_world, dtype=np.float64)

class BirdEyeParam:
    def __init__(self, x0=0.6, dx=1.2, dy=1.2, w=1024):
        # coordinates of viewing area on local floorplane in base_footprint frame
        #self.x0, self.dx, self.dy = 0.3, 1.5, 2.0 # breaks in reality? works in gazebo
        self.x0, self.dx, self.dy = x0, dx, dy#0.6, 1.2, 1.2
        
        #w = 1280; h = int(dx/dy*w)
        self.w = w; self.h = int(self.dx/self.dy*self.w)
        # viewing area in base_footprint frame
        # bottom_right, top_right, top_left, bottom_left in base_footprint frame
        self.va_bf = np.array([(self.x0, self.dy/2, 0.), (self.x0+self.dx, self.dy/2, 0.), (self.x0+self.dx, -self.dy/2, 0.), (self.x0, -self.dy/2, 0.)])

class CarolineBirdEyeParam(BirdEyeParam):
    def __init__(self, x0=0.2, dx=1.2, dy=0.6, w=480):
        BirdEyeParam.__init__(self, x0, dx, dy, w)

        
class BirdEyeTransformer:
    def __init__(self, cam, be_param):
        self.set_param(cam, be_param)
        self.cnt_fp = None
        self.unwarped_img = None
        
    def set_param(self, cam, be_param):
        self.param = be_param
        self.w, self.h = be_param.w, be_param.h
        va_img = cam.project(be_param.va_bf)
        self.va_img_undistorted = cam.undistort_points(va_img).astype(int).squeeze()
        pts_src = self.va_img_undistorted
        pts_dst = np.array([[0, self.h], [0, 0], [self.w, 0], [self.w, self.h]])
        self.H, status = cv2.findHomography(pts_src, pts_dst)
        print('calibrated bird eye: {} {}'.format(status, self.H))
        
    def plot_calibration(self, img_undistorted):
        pass
        
    def process(self, img):
        self.unwarped_img = cv2.warpPerspective(img, self.H, (self.w, self.h))
        return self.unwarped_img

    def draw_debug(self, cam, img=None, lane_model=None, cnts_be=None, border_color=128, fill_color=(255, 0, 255)):
        if img is None:
            img = np.zeros((self.h, self.w, 3), dtype=np.float32) # black image in be coordinates
        elif img.dtype == np.uint8:
            img = img.astype(np.float32)/255.
        if  cnts_be is not None:
            for cnt_be in cnts_be:
                cnt_be_int = cnt_be.astype(np.int32)
                cv2.drawContours(img, cnt_be_int, -1, (255,0,255), 3)
                cv2.fillPoly(img, pts =[cnt_be_int], color=fill_color)
        if lane_model is not None and lane_model.is_valid():
            self.draw_line(cam, img, lane_model, lane_model.x_min, lane_model.x_max)
        out_img = change_canvas(img, cam.h, cam.w)
        return out_img

    def draw_line(self, cam, img, lane_model, l0=0.6, l1=1.8, color=(0,128,0)):
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


    def points_imp_to_be(self, points_imp):
        return cv2.perspectiveTransform(points_imp, self.H)

    def unwarped_to_fp(self, cam, cnt_uw):
        #pdb.set_trace()
        s = self.param.dy/self.param.w
        self.cnt_fp = np.array([((self.param.h-p[1])*s+self.param.x0, (self.param.w/2-p[0])*s, 0.) for p in cnt_uw.squeeze()])
        return self.cnt_fp

    def lfp_to_unwarped(self, cam, cnt_lfp):
        s = self.param.dy/self.param.w
        cnt_uv = np.array([(self.param.w/2-y/s, self.param.h-(x-self.param.x0)/s) for x, y, _ in cnt_lfp])
        return cnt_uv



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
        self.cnt_max = None
        self.cnt_max_area = 0

    def has_contour(self): return (self.cnt_max is not None)
    def get_contour(self): return self.cnt_max
    def get_contour_area(self): return self.cnt_max_area
    
    def process(self, img):
        self.img2, self.cnts, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        self.cnt_max = None
        if self.cnts is not None and len(self.cnts) > 0:
            self.cnt_max = max(self.cnts, key=cv2.contourArea)
            self.cnt_max_area = cv2.contourArea(self.cnt_max)
            if self.min_area is not None and self.cnt_max_area < self.min_area:
               self.cnt_max, self.cnt_max_area = None, 0
            
            if not self.single_contour:
                self.cnt_areas = np.array([cv2.contourArea(c) for c in self.cnts])
                #self.cnt_areas_order = np.argsort(self.cnt_areas)
                self.valid_cnts_idx = self.cnt_areas > self.min_area
                self.valid_cnts = np.array(self.cnts)[self.valid_cnts_idx]
               
               
    def draw(self, img, color=(255,0,0), thickness=3, fill=True, fill_color=(255,0,0), draw_all=False):
        if self.cnt_max is not None:
            if fill:
                try:
                    cv2.fillPoly(img, pts =[self.cnt_max], color=fill_color)
                except cv2.error: # fails when poly is too small?
                    #print self.cnt_max.shape
                    pass
            cv2.drawContours(img, self.cnt_max, -1, color, thickness)
        if draw_all and self.valid_cnts is not None:
            for c in self.valid_cnts:
                cv2.fillPoly(img, pts =[c], color=(0, 128, 128))
            

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
        self.bin_ctr_finder.draw(bgr_img, thickness=1, color=color, fill_color=fill_color)
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
        ret, self.threshold = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)
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
    def __init__(self):
        self.img, self.lines, self.vlines = None, None, None
        y0, y1, y2 = 850, 950, 1000
        x1, x2, x3 = 110, 350, 480 
        self.mask_be = np.array([[(0, y0), (x1, y1), (x2, y1), (x3, y0), (x3, y2), (0, y2)]])
        self.use_mask=True
        
    def process(self, img, minval=100, maxval=200):
        self.edges = cv2.Canny(img, minval, maxval, apertureSize=3)
        if self.use_mask: cv2.fillPoly(self.edges, pts =self.mask_be, color=(0))
        rho_acc, theta_acc = 6, np.deg2rad(1)
        lines, threshold = 80, 30
        minLineLength, maxLineGap = 50, 25
        self.lines = cv2.HoughLinesP(self.edges, rho=rho_acc, theta=theta_acc , threshold=threshold,
                                     lines=lines, minLineLength=minLineLength, maxLineGap=maxLineGap)
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
        
    def draw(self, img, color=(0,0,255)):
        for line in self.lines:
            x1,y1,x2,y2 = line.squeeze()
            cv2.line(img,(x1,y1),(x2,y2),color,2)
        for line in self.vlines:
            x1,y1,x2,y2 = line.squeeze()
            cv2.line(img,(x1,y1),(x2,y2),(255, 0, 0),2)
        cv2.polylines(img, self.mask_be, isClosed=True, color=(0, 255, 255), thickness=2)

            
def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])

class FloorPlaneInjector:
    def __init__(self):
        self.contour_floor_plane_blf = None

    def compute(self, contour, cam):
        # undistorted coordinates
        contour_undistorted = cv2.undistortPoints(contour.astype(np.float32), cam.K, cam.D, None, cam.K)
        # contour in optical plan
        contour_cam = [np.dot(cam.invK, [u, v, 1]) for (u, v) in contour_undistorted.squeeze()]
        # contour projected on floor plane (in cam frame)
        contour_floor_plane_cam = get_points_on_plane(contour_cam, cam.fp_n, cam.fp_d)
        # contour projected on floor plane (in body frame)
        self.contour_floor_plane_blf = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in contour_floor_plane_cam])

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

    def draw_timing(self, img, x0=280, y0=20, dy=35, h=0.75, color_bgr=(220, 220, 50)):
        f, c, w = cv2.FONT_HERSHEY_SIMPLEX, color_bgr, 2
        try: 
            txt = 'fps: {:.1f} (min {:.1f} max {:.1f})'.format(self.lp_fps, self.min_fps, self.max_fps)
            cv2.putText(img, txt, (x0, y0), f, h, c, w)
            txt = 'skipped: {:d} (cpu {:.3f}/{:.3f}s)'.format(self.skipped_frames, self.lp_proc, 1./self.lp_fps)
            cv2.putText(img, txt, (x0, y0+dy), f, h, c, w)
        except AttributeError: pass
        






 
