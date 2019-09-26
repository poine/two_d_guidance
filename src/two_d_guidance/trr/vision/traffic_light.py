import numpy as np
import cv2
import PIL, PIL.ImageFont, PIL.ImageDraw

import two_d_guidance.trr.vision.utils as trr_vu
import pdb

class TrafficLightPipeline(trr_vu.Pipeline):
    show_none, show_input, show_roi, show_green_mask, show_red_mask, show_green_blob, show_red_blob, show_contours = range(8)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.img_bgr = None
        self.green_ctr_detc = trr_vu.ColoredContourDetector(trr_vu.hsv_green_range(), min_area=10)
        self.red_ctr_detc = trr_vu.ColoredContourDetector(trr_vu.hsv_red_range(), min_area=10)
        cfg_path = "/home/poine/work/roverboard/roverboard_caroline/config/traffic_light_blob_detector_cfg.yaml"
        self.green_blob_detc = trr_vu.ColoredBlobDetector(trr_vu.hsv_green_range(), cfg_path)
        self.red_blob_detc = trr_vu.ColoredBlobDetector(trr_vu.hsv_red_range(), cfg_path)
        
        self.set_roi((150, 15), (300, 100))
        self.set_debug_display(TrafficLightPipeline.show_none, False)
        self.pil_font = PIL.ImageFont.truetype("FreeMono.ttf", 30)  

    def set_debug_display(self, display_mode, show_hud):
        self.display_mode, self.show_hud = display_mode, show_hud
        
    def set_roi(self, tl, br):
        print('roi: {}'.format(tl, br))
        self.roi_tl, self.roi_br = tl, br
        self.roi_h, self.roi_w = br[1]-tl[1], br[0]-tl[0]
        self.roi = slice(tl[1], br[1]), slice(tl[0], br[0])
        
    def set_green_mask_params(self, hc, hs, smin, smax, vmin, vmax): 
        self.green_ctr_detc.set_hsv_range(trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax))

    def set_red_mask_params(self, hc, hs, smin, smax, vmin, vmax): 
        red_range = trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax)
        #print('red range : {}'.format(red_range))
        self.red_ctr_detc.set_hsv_range(red_range)
        
    def _process_image(self, img_bgr, cam, stamp):
        # we receive a BGR image 
        self.img_bgr = img_bgr
        #self.img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Extract Region of Interest
        self.roi_img_bgr = self.img_bgr[self.roi]
        # Convert it to HVS
        self.roi_img_hsv = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2HSV)
        self.roi_img_gray = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2GRAY)
        # Feed it to contour detectors
        self.red_ctr_detc.process_hsv_image(self.roi_img_hsv, self.roi_img_gray)
        self.green_ctr_detc.process_hsv_image(self.roi_img_hsv, self.roi_img_gray)
        # and blob detectors
        #self.green_blob_detc.process_hsv_image(self.roi_img_hsv)
        #self.red_blob_detc.process_hsv_image(self.roi_img_hsv)
        
    def sees_red(self): return self.red_ctr_detc.has_contour()
    def sees_green(self): return self.green_ctr_detc.has_contour()
    def get_light_status(self): return self.sees_red(), False, self.sees_green()
    
    def draw_debug(self, cam, img_cam=None, scale_roi=True, border_color=128):
        if self.img_bgr is None: # we have nothing to display, return a black image
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == TrafficLightPipeline.show_input:
            out_img = self.img_bgr
            cv2.rectangle(out_img, tuple(self.roi_tl), tuple(self.roi_br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == TrafficLightPipeline.show_roi:
            out_roi = self.roi_img_bgr
        elif self.display_mode == TrafficLightPipeline.show_green_mask:
            out_roi = cv2.cvtColor(self.green_ctr_detc.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == TrafficLightPipeline.show_red_mask:
            out_roi = cv2.cvtColor(self.red_ctr_detc.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == TrafficLightPipeline.show_green_blob:
            out_roi = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2GRAY)
            out_roi = cv2.cvtColor(out_roi, cv2.COLOR_GRAY2BGR)
            self.green_blob_detc.draw(out_roi, color=(0, 255, 0))
            print('keypointds green {}'.format(self.green_blob_detc.keypoints_nb()))
        elif self.display_mode == TrafficLightPipeline.show_red_blob:
            out_roi = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2GRAY)
            out_roi = cv2.cvtColor(out_roi, cv2.COLOR_GRAY2BGR)
            self.red_blob_detc.draw(out_roi, color=(0, 0, 255))
            print('keypointds red {}'.format(self.red_blob_detc.keypoints_nb()))
        elif self.display_mode == TrafficLightPipeline.show_contours:
            out_roi = cv2.cvtColor(self.roi_img_bgr, cv2.COLOR_BGR2GRAY)
            out_roi = cv2.cvtColor(out_roi, cv2.COLOR_GRAY2BGR)
            self.green_ctr_detc.draw(out_roi, color=(155, 155, 0), fill_color=(0, 255,0))
            self.red_ctr_detc.draw(out_roi, color=(155, 155, 0), fill_color=(0,0,255))
        # display the region of interest as thumbmnail or scaled
        if self.display_mode != TrafficLightPipeline.show_input: # in all other modes, we work on a roi
            out_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            if not scale_roi:
                out_img[self.roi] = out_roi
            else:
                scale = min(float(cam.h)/self.roi_h, float(cam.w)/self.roi_w)
                h, w = int(scale*self.roi_h), int(scale*self.roi_w)
                dx, dy = (cam.w-w)/2, (cam.h-h)/2
                out_img[dy:dy+h, dx:dx+w] = cv2.resize(out_roi, (w, h))
            
        if self.show_hud: out_img = self.draw_hud(out_img)
        # we return a RGB image
        return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)


    def draw_hud(self, out_img, font_color=(0,0,255)): # red
        self.draw_timing(out_img, y0=30)
        sta_red =   'red   : area: {}'.format(self.red_ctr_detc.get_contour_area()) if self.sees_red()     else 'red:   not detected'
        sta_green = 'green : area: {}'.format(self.green_ctr_detc.get_contour_area()) if self.sees_green() else 'green: not detected'
        if 0:
            # this is blue, so out_img is BGR
            f, h, c, w, y0 = cv2.FONT_HERSHEY_PLAIN, 1.25, font_color, 2
            cv2.putText(out_img, 'Traffic light', (y0, 40), f, h, c, w)
            cv2.putText(out_img, sta_red, (y0, 390), f, h, c, w)
            cv2.putText(out_img, sta_green, (y0, 440), f, h, c, w)
        else:
            im_pil = PIL.Image.fromarray(out_img)
            draw = PIL.ImageDraw.Draw(im_pil)
            #font_color = (255,0,0) # blue
            font_color = (0,0,255) # red
            draw.text((10, 10), 'Traffic light', font=self.pil_font, fill=font_color)
            draw.text((10, 410), sta_red, font=self.pil_font, fill=font_color)
            draw.text((10, 440), sta_green, font=self.pil_font, fill=font_color)
            out_img = np.array(im_pil)
        return out_img
    
