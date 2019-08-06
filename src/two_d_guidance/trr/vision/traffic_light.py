import numpy as np
import cv2

import two_d_guidance.trr_vision_utils as trr_vu
import pdb

class TrafficLightPipeline(trr_vu.Pipeline):
    show_none, show_roi, show_green_mask, show_red_mask, show_contour = range(5)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.img_bgr = None
        self.green_ctr_detc = trr_vu.ColoredContourDetector(trr_vu.hsv_green_range(), min_area=10)
        self.red_ctr_detc = trr_vu.ColoredContourDetector(trr_vu.hsv_red_ranges(), min_area=10)
        self.set_roi((150, 15), (300, 100))
        self.display_mode = TrafficLightPipeline.show_none

    def set_roi(self, tl, br):
        print('roi: {}'.format(tl, br))
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        
    def set_green_mask_params(self, hsens, smin, smax, vmin, vmax): 
        green_range = [[np.array([60 - hsens, smin, vmin]), np.array([60 + hsens, smax, vmax])]]
        print('green range: {}'.format(green_range))
        self.green_ctr_detc.set_hsv_ranges(green_range)

    def set_red_mask_params(self, hsens, smin, smax, vmin, vmax): 
        red_ranges = [[np.array([0,smin,vmin]),   np.array([hsens,smax,vmax])],
                      [np.array([180-hsens,smin,vmin]), np.array([180,smax,vmax])]]
        print('red ranges : {}'.format(red_ranges))
        self.red_ctr_detc.set_hsv_ranges(red_ranges)

        
    def _process_image(self, img_bgr, cam):
        # we receive a BGR image from gazebo 
        #self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # we receive a BGR image from gazebo
        self.img_bgr = img_bgr
        #w, h = br - tl
        #tr = tl + [w, 0]
        #bl = br - [w, 0]
        #vertices = np.array([tl, tr, br, bl], dtype=np.float32)
        #self.roi_img = self.img[120:230,15:92]
        #mask = np.zeros_like(img)
        #mask = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        #cv2.rectangle(mask, tuple(tl), tuple(br), color=(255, 255, 255), thickness=-1)#cv2.CV_FILLED)
        #cv2.fillPoly(mask, vertices, 255)
        #self.masked_image = cv2.bitwise_and(self.img_bgr, mask)
        self.roi_img = self.img_bgr[self.roi]
        hsv_image = cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2HSV)
        #hsv_image = cv2.cvtColor(self.roi_img, cv2.COLOR_RGB2HSV)
        self.red_ctr_detc.process_hsv_image(hsv_image)
        self.green_ctr_detc.process_hsv_image(hsv_image)
        
    def draw_debug(self, cam, img_cam=None, scale_roi=True, border_color=128):
        if self.img_bgr is None: # we have nothing to display, return a black image
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == TrafficLightPipeline.show_roi:
            out_img = self.img_bgr
            cv2.rectangle(out_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == TrafficLightPipeline.show_green_mask:
            out_roi = cv2.cvtColor(self.green_ctr_detc.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == TrafficLightPipeline.show_red_mask:
            out_roi = cv2.cvtColor(self.red_ctr_detc.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == TrafficLightPipeline.show_contour:
            out_roi = cv2.cvtColor(self.roi_img, cv2.COLOR_BGR2GRAY)
            out_roi = cv2.cvtColor(out_roi, cv2.COLOR_GRAY2BGR)
            self.green_ctr_detc.draw(out_roi, color=(155, 155, 0), fill_color=(0, 255,0))
            self.red_ctr_detc.draw(out_roi, color=(155, 155, 0), fill_color=(0,0,255))
        if self.display_mode != TrafficLightPipeline.show_roi: # in all other modes, we work on a roi
            out_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            if not scale_roi:
                out_img[self.roi] = out_roi
            else:
                scale = min(float(cam.h)/self.roi_h, float(cam.w)/self.roi_w)
                h, w = int(scale*self.roi_h), int(scale*self.roi_w)
                dx, dy = (cam.w-w)/2, (cam.h-h)/2
                out_img[dy:dy+h, dx:dx+w] = cv2.resize(out_roi, (w, h))
            
        # this is blue, so out_img is BGR
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        cv2.putText(out_img, 'traffic light', (20, 40), f, h, c, w)
        self.draw_timing(out_img, y0=90)
        sta_green = 'green: {} area: {}'.format(self.green_ctr_detc.has_contour(), self.green_ctr_detc.get_contour_area())
        cv2.putText(out_img, sta_green, (20, 240), f, h, c, w)
        sta_red = 'red: {} area: {}'.format(self.red_ctr_detc.has_contour(), self.red_ctr_detc.get_contour_area())
        cv2.putText(out_img, sta_red, (20, 290), f, h, c, w)
        # we return a RGB image
        return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
