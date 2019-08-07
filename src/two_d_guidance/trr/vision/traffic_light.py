import numpy as np
import cv2

import two_d_guidance.trr_vision_utils as trr_vu
import pdb

class TrafficLightPipeline(trr_vu.Pipeline):
    show_none, show_input, show_roi, show_green_mask, show_red_mask, show_contour = range(6)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.img_bgr = None
        self.green_ctr_detc = trr_vu.ColoredContourDetector(trr_vu.hsv_green_range(), min_area=10)
        self.red_ctr_detc = trr_vu.ColoredContourDetector(trr_vu.hsv_red_ranges(), min_area=10)
        cfg_path = "/home/poine/work/roverboard/roverboard_caroline/config/traffic_light_blob_detector_cfg.yaml"
        self.green_blob_detc = trr_vu.ColoredBlobDetector(trr_vu.hsv_green_range(), cfg_path)
        self.set_roi((150, 15), (300, 100))
        self.set_debug_display(TrafficLightPipeline.show_none, False)

    def set_debug_display(self, display_mode, show_hud):
        self.display_mode, self.show_hud = display_mode, show_hud
        
    def set_roi(self, tl, br):
        print('roi: {}'.format(tl, br))
        self.roi_tl, self.roi_br = tl, br
        self.roi_h, self.roi_w = br[1]-tl[1], br[0]-tl[0]
        self.roi = slice(tl[1], br[1]), slice(tl[0], br[0])
        
    def set_green_mask_params(self, hc, hs, smin, smax, vmin, vmax): 
        self.green_ctr_detc.set_hsv_ranges(trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax))

    def set_red_mask_params(self, hc, hs, smin, smax, vmin, vmax): 
        red_ranges = trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax)
        #print('red ranges : {}'.format(red_ranges))
        self.red_ctr_detc.set_hsv_ranges(red_ranges)

        
    def _process_image(self, img_bgr, cam):
        # we receive a BGR image 
        self.img_bgr = img_bgr
        #self.img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Extract Region of Interest
        self.roi_img_bgr = self.img_bgr[self.roi]
        # Convert it to HVS
        self.roi_img_hsv = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2HSV)
        # Feed it to contour detectors
        self.red_ctr_detc.process_hsv_image(self.roi_img_hsv)
        self.green_ctr_detc.process_hsv_image(self.roi_img_hsv)
        self.green_blob_detc.process_hsv_image(self.roi_img_hsv)
        
    def draw_debug(self, cam, img_cam=None, scale_roi=True, border_color=128):
        if self.img_bgr is None: # we have nothing to display, return a black image
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == TrafficLightPipeline.show_input:
            out_img = self.img_bgr
            cv2.rectangle(out_img, tuple(self.roi_tl), tuple(self.roi_br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == TrafficLightPipeline.show_roi:
            out_roi = self.roi_img_bgr
            #cv2.rectangle(out_img, (self.px-1, self.py-1), (self.px+1, self.py+1), color=(0, 0, 255), thickness=3)
        elif self.display_mode == TrafficLightPipeline.show_green_mask:
            out_roi = cv2.cvtColor(self.green_ctr_detc.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == TrafficLightPipeline.show_red_mask:
            out_roi = cv2.cvtColor(self.red_ctr_detc.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == TrafficLightPipeline.show_contour:
            out_roi = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2GRAY)
            out_roi = cv2.cvtColor(out_roi, cv2.COLOR_GRAY2BGR)
            #self.green_ctr_detc.draw(out_roi, color=(155, 155, 0), fill_color=(0, 255,0))
            self.red_ctr_detc.draw(out_roi, color=(155, 155, 0), fill_color=(0,0,255))
            self.green_blob_detc.draw(out_roi)
        if self.display_mode != TrafficLightPipeline.show_input: # in all other modes, we work on a roi
            out_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            if not scale_roi:
                out_img[self.roi] = out_roi
            else:
                scale = min(float(cam.h)/self.roi_h, float(cam.w)/self.roi_w)
                h, w = int(scale*self.roi_h), int(scale*self.roi_w)
                dx, dy = (cam.w-w)/2, (cam.h-h)/2
                out_img[dy:dy+h, dx:dx+w] = cv2.resize(out_roi, (w, h))
            
        if self.show_hud:
            # this is blue, so out_img is BGR
            f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
            cv2.putText(out_img, 'Traffic light', (20, 40), f, h, c, w)
            self.draw_timing(out_img, y0=90)
            sta_green = 'green: {} area: {}'.format(self.green_ctr_detc.has_contour(), self.green_ctr_detc.get_contour_area())
            cv2.putText(out_img, sta_green, (20, 390), f, h, c, w)
            sta_red = 'red: {} area: {}'.format(self.red_ctr_detc.has_contour(), self.red_ctr_detc.get_contour_area())
            cv2.putText(out_img, sta_red, (20, 440), f, h, c, w)
        # we return a RGB image
        return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
