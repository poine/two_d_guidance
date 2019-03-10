#!/usr/bin/env python
import time, math, numpy as np

import cv2
import pdb


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    #cv2.fillPoly(mask, np.array([[(0, 0), (100, 100), (200,0)]], dtype=np.int32), 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest_vertices(height, width):
    return np.array([[
    (0, height),
    (0.3*width, 0.2*height),
    (0.7*width, 0.2*height),
    (width, height),
    ]], dtype=np.int32)

class LaneFinder:

    def __init__(self):
        pass


    def process_image(self, img, cam_idx):
        self.inputImageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.inputImageGray,100,200,apertureSize = 3)
        h, w, depth = img.shape
        self.roi = region_of_interest_vertices(h, w)
        self.edges_cropped = region_of_interest(self.edges, self.roi)
        minLineLength = 25 
        maxLineGap = 25 
        self.lines = cv2.HoughLinesP(self.edges_cropped, rho=6, theta=np.pi/180 , threshold=30, lines=90, minLineLength=minLineLength, maxLineGap=maxLineGap)


    def draw(self, img, cam):
        out_img = cv2.cvtColor(self.inputImageGray, cv2.COLOR_GRAY2BGR)
        if 1:
            cv2.polylines(out_img, self.roi, isClosed=True, color=(255, 0, 0), thickness=2)
        if 1:
            #try:
            for [[x1,y1,x2,y2]] in self.lines:
                cv2.line(out_img,(x1,y1),(x2,y2),(0,128,0),2)
            #except TypeError: pass
        return out_img

if __name__ == '__main__':
    img = cv2.imread('/home/poine/work/robot_data/oscar/gazebo_lines/image_01.png', cv2.IMREAD_UNCHANGED)
    lane_finder = LaneFinder()
    lane_finder.draw(img, None)
    
    cv2.imshow('follow line', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
