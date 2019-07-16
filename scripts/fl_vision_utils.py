#!/usr/bin/env python
import time, math, numpy as np

import cv2
from matplotlib import pyplot as plt
import yaml

import fl_utils as flu, smocap
import pdb

def load_cam_from_files(intr_path, extr_path, cam_name='cam1', alpha=1.):
    cam = smocap.Camera(0, 'cam1')
    cam.load_intrinsics(intr_path)
    cam.set_undistortion_param(alpha)
    cam.load_extrinsics(extr_path)
    return cam


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
    def __init__(self, cam, param):
        self.param = param
        self.calibrate(cam, param)
        self.cnt_fp = None
        self.unwarped_img = None
        
    def calibrate(self, cam, param):
        self.w, self.h = param.w, param.h
        va_img = cam.project(param.va_bf)
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

    def draw_debug(self, cam, img=None, lane_model=None, border_color=0.4):
        if img is None: img = self.unwarped_img
        if img.dtype == np.uint8:
            img = img.astype(np.float32)/255.
        out_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.float32)
        scale = min(float(cam.h)/self.h, float(cam.w)/self.w)
        h, w = int(scale*self.h), int(scale*self.w)
        dx = (cam.w-w)/2
        if lane_model is not None and lane_model.is_valid(): self.draw_line(cam, img, lane_model, lane_model.x_min, lane_model.x_max)
        out_img[:h, dx:dx+w] = cv2.resize(img, (w, h))
        return out_img

    def draw_line(self, cam, img, lane_model, l0=0.6, l1=1.8):
        # Coordinates in local_floor_plane(aka base_link_footprint) frame
        xs = np.linspace(l0, l1, 20); ys = lane_model.get_y(xs)
        pts_lfp = np.array([[x, y, 0] for x, y in zip(xs, ys)])
        pts_img = self.lfp_to_unwarped(cam, pts_lfp)
        #pdb.set_trace()
        for i in range(len(pts_img)-1):
            try:
                #print(tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)))
                cv2.line(img, tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)), (0,128,0), 4)
            except OverflowError:
                pass

    
    def unwarped_to_fp(self, cam, cnt_uw):
        #pdb.set_trace()
        s = self.param.dy/self.param.w
        self.cnt_fp = np.array([((self.param.h-p[1])*s+self.param.x0, (self.param.w/2-p[0])*s, 0.) for p in cnt_uw.squeeze()])


    def lfp_to_unwarped(self, cam, cnt_lfp):
        s = self.param.dy/self.param.w
        cnt_uv = np.array([(self.param.w/2-y/s, self.param.h-(x-self.param.x0)/s) for x, y, _ in cnt_lfp])
        return cnt_uv

    
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    #cv2.fillPoly(mask, np.array([[(0, 0), (100, 100), (200,0)]], dtype=np.int32), 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest_vertices(height, width, dh=0.25):
    return np.array([[
    (0, height),
    (0.4*width, dh*height),
    (0.6*width, dh*height),
    (width, height),
    ]], dtype=np.int32)


class BinaryThresholder:
    def __init__(self, thresh=196):
        self.thresh_val = thresh#130#196
        self.threshold = None
        
    def process(self, img):
        #pdb.set_trace()
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        ret, self.threshold = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)
        return self.threshold

    def set_threshold(self, thresh): self.thresh_val = thresh

    
class ContourFinder:
    def __init__(self):
        self.img = None
        self.cnt_be = None
        
    def process(self, img):
        self.img2, cnts, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #print cnts[0].shape
        self.cnt_be = None
        if cnts is not None and len(cnts) > 0:
            self.cnt_be = max(cnts, key=cv2.contourArea)
            
    def draw(self, img):
        if self.cnt_be is not None:
            cv2.drawContours(img, self.cnt_be, -1, (255,0,0), 3)

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
        self.img = None

    def process(self, img):
        self.edges = cv2.Canny(img, 100, 200, apertureSize = 3)
        minLineLength, maxLineGap = 25, 25 
        self.lines = cv2.HoughLinesP(self.edges, rho=6, theta=np.pi/180 , threshold=30, lines=90, minLineLength=minLineLength, maxLineGap=maxLineGap)
        print('found {} lines'.format(len(self.lines)))



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


class Pipeline:
    def __init__(self):
        self.skipped_frames = 0
        self.last_seq = None
        self.last_stamp = None
        self.real_fps = 0.
        
    def process_image(self, img, cam, stamp, seq):
        if self.last_stamp is not None:
            self.real_fps = 1./(stamp - self.last_stamp).to_sec()
        self.last_stamp = stamp
        if self.last_seq is not None:
            self.skipped_frames += seq-self.last_seq-1
        self.last_seq = seq
        _start = time.time()
        self._process_image(img, cam)
        _end = time.time()
        self.last_processing_duration = _end-_start

    def draw_timing(self, img):
        cv2.putText(img, 'fps: {:.1f}'.format(self.real_fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(img, 'skipped: {:d}'.format(self.skipped_frames), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        try: cv2.putText(img, 'proc fps: {:.1f}'.format(1./self.last_processing_duration), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        except AttributeError: pass
        
        
        
class Contour1Pipeline(Pipeline):
    
    def __init__(self, cam):
        Pipeline.__init__(self)
        self.thresholder = BinaryThresholder()
        self.contour_finder = ContourFinder()
        self.floor_plane_injector = FloorPlaneInjector()
        self.lane_model = flu.LaneModel()
        self.display_mode = 0

    def _process_image(self, img, cam):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.thresholder.process(self.img_gray)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_be is not None:
            self.floor_plane_injector.compute(self.contour_finder.cnt_be, cam)
            self.lane_model.fit(self.floor_plane_injector.contour_floor_plane_blf[:,:2])
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
        
    def draw_debug(self, cam, img_cam=None):
        out_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.draw_timing(out_img)
        self.contour_finder.draw(out_img)
        if self.lane_model.is_valid():
            self.lane_model.draw_on_cam_img(out_img, cam)
        return out_img
        
class Contour2Pipeline(Pipeline):
    show_be, show_thresh, show_contour = range(3)
    def __init__(self, cam, be_param=BirdEyeParam()):
        self.bird_eye = BirdEyeTransformer(cam, be_param)
        self.thresholder = BinaryThresholder()
        self.contour_finder = ContourFinder()
        self.lane_model = flu.LaneModel()
        self.display_mode = Contour2Pipeline.show_be
         
    def _process_image(self, img, cam):
        undistorted_img = cam.undistort_img(img) # costly (0.05s)
        #undistorted_img = img
        # Project image to Bird Eye view
        self.bird_eye.process(undistorted_img)
        # Compute grayscale threshold
        self.thresholder.process(cv2.cvtColor(self.bird_eye.unwarped_img, cv2.COLOR_BGR2GRAY))
        # Find largest white contour in bird eye image
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_be is not None:
            # Project contour from bird_eye image to floor_plan
            try: # FIXME
                self.bird_eye.unwarped_to_fp(cam, self.contour_finder.cnt_be)
            except IndexError:
                #pdb.set_trace()
                pass
            # Fit lane model from contour (in floorplan(aka base_link_footprint frame)
            self.lane_model.fit(self.bird_eye.cnt_fp[:,:2])
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
        
    def draw_debug(self, cam, img_cam=None):
        if self.bird_eye.unwarped_img is None: return np.zeros((cam.h, cam.w, 3))
        if self.display_mode == Contour2Pipeline.show_be:
            img = self.bird_eye.unwarped_img
            self.contour_finder.draw(img)
            return self.bird_eye.draw_debug(cam, img, self.lane_model)
        elif self.display_mode == Contour2Pipeline.show_thresh:
            img =  cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            return self.bird_eye.draw_debug(cam, img, self.lane_model)
        elif self.display_mode == Contour2Pipeline.show_contour:
            img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.contour_finder.draw(img)
            return self.bird_eye.draw_debug(cam, img, self.lane_model)


class Foo3Pipeline(Pipeline):
    def __init__(self, cam, be_param=BirdEyeParam()):
        self.bird_eye = BirdEyeTransformer(cam, be_param)
        self.lane_model = flu.LaneModel()
        self.undistorted_img = None
        
    def _process_image(self, img, cam):
        self.undistorted_img = cam.undistort_img(img)
        self.bird_eye.process(self.undistorted_img)
        self.edges = cv2.Canny(self.bird_eye.unwarped_img, 100, 200)
        
    def draw_debug(self, cam, img_cam=None):
        if 0: img_out = img_cam
        if 0: img_out = self.undistorted_img if self.undistorted_img is not None else np.zeros((cam.h, cam.w, 3))
        if 0: img_out = self.bird_eye.draw_debug(cam, self.bird_eye.unwarped_img)
        if 1: img_out = self.bird_eye.draw_debug(cam, cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR))
        #if self.undistorted_img is not None:
        #    print(self.undistorted_img.dtype, self.bird_eye.unwarped_img.dtype)
        return img_out

    
class LaneFinder:

    def __init__(self, cam):
       
        #self.houghlinesfinder = HoughLinesFinder()
        self.c1pipe = Contour1Pipeline(cam)
        self.c2pipe = Contour2Pipeline(cam)
        self.inputImageGray = None


    def process_rgb_image(self, img, cam):
        self.c2pipe.process_image(img, cam)
        #self.process_mono_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam)
        
    def process_mono_image(self, img, cam):
        self.inputImageGray = img
        self.c1pipe.process_image(img, cam)
       #self.houghlinesfinder.process(img)
        #pdb.set_trace()
       
        #self.find_contour(self.bird_eye.unwarped_img)
        #self.find_lines(img)

    def find_lines(self, img):
        self.inputImageGray = img
        self.edges = cv2.Canny(self.inputImageGray, 100, 200, apertureSize = 3)
        h, w = img.shape
        self.roi = region_of_interest_vertices(h, w)
        self.edges_cropped = region_of_interest(self.edges, self.roi)
        minLineLength = 25 
        maxLineGap = 25 
        self.lines = cv2.HoughLinesP(self.edges_cropped, rho=6, theta=np.pi/180 , threshold=30, lines=90, minLineLength=minLineLength, maxLineGap=maxLineGap)


    def draw(self, img, cam):
        #if self.inputImageGray is None: return np.zeros((cam.h, cam.w, 3))
        if True: # pipe2: birdeye
            out_img = self.c2pipe.draw_debug(cam)
        if False: # pipe1: direct
            out_img = self.c1pipe.draw_debug(cam)
        if False and self.houghlinesfinder.lines is not None:
            #try:
            out_img = cv2.cvtColor(self.houghlinesfinder.edges, cv2.COLOR_GRAY2BGR)
            for [[x1,y1,x2,y2]] in self.houghlinesfinder.lines:
                cv2.line(out_img,(x1,y1),(x2,y2),(0,128,0),2)
            #except TypeError: pass
        if False:
            #cv2.drawContours(out_img, self.cnt_be, -1, (0,0,255), 3)
            cont_img_orig = cv2.perspectiveTransform(self.cnt_be.astype(np.float32), self.bird_eye.h_inv)
            
        return out_img



def plot(lf):
    
    #plt.subplot(221), plt.imshow(lf.inputImageGray, cmap = 'gray')
    #plt.title('camera')
    #plt.subplot(222), plt.imshow(lf.edges, cmap = 'gray')
    #plt.title('edges')
    be_calib_img = cv2.cvtColor(lf.inputImageGray, cv2.COLOR_GRAY2BGR)
    be_calib_img = lf.bird_eye.plot_calibration(be_calib_img)
    plt.subplot(221), plt.imshow(be_calib_img)
    
    plt.subplot(222), plt.imshow(lf.bird_eye.unwarped_img, cmap = 'gray')
    plt.title('bird eye')
    
    plt.subplot(223), plt.imshow(lf.edges_cropped, cmap = 'gray')
    plt.title('edges cropped')
    result_img = lane_finder.draw(lf.inputImageGray, None)
    plt.subplot(224), plt.imshow(result_img, cmap = 'gray')
    plt.title('result')
    plt.show()
    
    
if __name__ == '__main__':
    #img = cv2.imread('/home/poine/work/robot_data/oscar/gazebo_lines/image_01.png', cv2.IMREAD_UNCHANGED)
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray =  cv2.imread('/home/poine/work/robot_data/oscar/gazebo_lines/image_07.png', cv2.IMREAD_GRAYSCALE)
    img_gray =  cv2.imread('/home/poine/work/robot_data/jeanmarie/image_02.png', cv2.IMREAD_GRAYSCALE)
    
    lane_finder = LaneFinder()
    lane_finder.process_image(img_gray, 0)

    plot(lane_finder)
    #cv2.imshow('follow line', img_gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
