#!/usr/bin/env python
import time, math, numpy as np

import cv2
from matplotlib import pyplot as plt

import pdb

class BirdEyeTransformer:
    def __init__(self):
        self.foo = None
        top_x, top_y, bottom_x, bottom_y = 80, 60, 340, 75
        self.pt0x, self.pt0y = 320, 200; self.pt0 = (self.pt0x, self.pt0y)
        self.pb0x, self.pb0y = 320, 280; self.pb0 = (self.pb0x, self.pb0y)
        
        self.pts_src = np.array([[self.pt0x - top_x, self.pt0y - top_y],
                                 [self.pt0x + top_x, self.pt0y - top_y],
                                 [self.pb0x + bottom_x, self.pb0y + bottom_y],
                                 [self.pb0x - bottom_x, self.pb0y + bottom_y]])
        self.pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

        self.calibrate()

    def calibrate(self):
        self.h, status = cv2.findHomography(self.pts_src, self.pts_dst)
        print('calibrated bird eye: {}'.format(status, self.h))
        self.h_inv = np.linalg.inv(self.h)
        
    def plot_calibration(self, img):
        lc, lw = (255, 0, 0), 1
        for p,t in zip([self.pt0, self.pb0], ['pt0', 'pb0']):
            cv2.circle(img,  p, 1, lc, -1); cv2.putText(img, t, p, cv2.FONT_HERSHEY_SIMPLEX, 1., lc, 1)
        for i in range(4):
            cv2.line(img, tuple(self.pts_src[i]), tuple(self.pts_src[i-1]), lc, lw)
            
        # cv2.line(img, tuple(self.pts_src[0]), tuple(self.pts_src[1]), lc, lw) # top line
        # cv2.line(img, tuple(self.pts_src[3]), tuple(self.pts_src[2]), lc, lw) # bottom line
        # cv2.line(img, tuple(self.pts_src[1]), tuple(self.pts_src[2]), lc, lw) # right line
        # cv2.line(img, tuple(self.pts_src[3]), tuple(self.pts_src[0]), lc, lw) # left_line
        return img
        
    def process(self, img_gray):
        self.unwarped_img = cv2.warpPerspective(img_gray, self.h, (640, 360))



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
    def __init__(self):
        self.low, self.high = 196, 255
        self.threshold = None
        
    def process(self, img):
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        ret, self.threshold = cv2.threshold(blurred, self.low, self.high, 0)
        
class HoughLinesFinder:
    def __init__(self):
        self.img = None

    def process(self, img):
        self.edges = cv2.Canny(img, 100, 200, apertureSize = 3)
        minLineLength, maxLineGap = 25, 25 
        self.lines = cv2.HoughLinesP(self.edges, rho=6, theta=np.pi/180 , threshold=30, lines=90, minLineLength=minLineLength, maxLineGap=maxLineGap)
        print('found {} lines'.format(len(self.lines)))

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


def get_points_on_plane(rays, plane_n, plane_d):
    return np.array([-plane_d/np.dot(ray, plane_n)*ray for ray in rays])

class FloorPlaneInjector:
    def __init__(self):
        self.contour_floor_plane_blf = None

    def compute(self, contour, cam):
        #pdb.set_trace()
        contour_undistorted = cv2.undistortPoints(contour.astype(np.float32), cam.K, cam.D, None, cam.K)
        #contour_undistorted = contour
        # contour in optical plan
        contour_cam = [np.dot(cam.invK, [u, v, 1]) for (u, v) in contour_undistorted.squeeze()]
        # contour projected on floor plane (in cam frame)
        contour_floor_plane_cam = get_points_on_plane(contour_cam, cam.fp_n, cam.fp_d)
        # contour projected on floor plane (in body frame)
        self.contour_floor_plane_blf = np.array([np.dot(cam.cam_to_world_T[:3], p.tolist()+[1]) for p in contour_floor_plane_cam])
            
class LaneFinder:

    def __init__(self):
        self.thresholder = BinaryThresholder()
        self.bird_eye = BirdEyeTransformer()
        #self.houghlinesfinder = HoughLinesFinder()
        self.contour_finder = ContourFinder()
        self.floor_plane_injector = FloorPlaneInjector()
        self.inputImageGray = None


    def process_rgb_image(self, img, cam):
        self.process_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam)
        
    def process_image(self, img, cam):
        self.inputImageGray = img
        self.thresholder.process(img)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_be is not None:
            self.floor_plane_injector.compute(self.contour_finder.cnt_be, cam)
        #self.houghlinesfinder.process(img)
        #pdb.set_trace()
        self.bird_eye.process(img)
        #self.find_contour(self.bird_eye.unwarped_img)
        #self.find_lines(img)

    def find_contour(self, img):
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        ret, thresh = cv2.threshold(blurred, 127, 255, 0)
        img2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #print cnts[0].shape
        self.cnt_be = None
        if cnts is not None and len(cnts) > 0:
            self.cnt_be = max(cnts, key=cv2.contourArea)
        #print('found contour {}'.format(self.cnt_be))
        
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
        if self.inputImageGray is None: return np.zeros((480, 640, 3))
        out_img = cv2.cvtColor(self.inputImageGray, cv2.COLOR_GRAY2BGR)
        if True:
            out_img = cv2.cvtColor(self.bird_eye.unwarped_img, cv2.COLOR_GRAY2BGR)

        if False:
            out_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        if False:
            cv2.polylines(out_img, self.roi, isClosed=True, color=(255, 0, 0), thickness=2)
        if False and self.houghlinesfinder.lines is not None:
            #try:
            out_img = cv2.cvtColor(self.houghlinesfinder.edges, cv2.COLOR_GRAY2BGR)
            for [[x1,y1,x2,y2]] in self.houghlinesfinder.lines:
                cv2.line(out_img,(x1,y1),(x2,y2),(0,128,0),2)
            #except TypeError: pass
        if False:
            #cv2.drawContours(out_img, self.cnt_be, -1, (0,0,255), 3)
            cont_img_orig = cv2.perspectiveTransform(self.cnt_be.astype(np.float32), self.bird_eye.h_inv)
        if False:
            #pdb.set_trace()
            #cv2.drawContours(out_img, cont_img_orig.astype(int), -1, (0,0,255), 3)
            self.contour_finder.draw(out_img)
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
