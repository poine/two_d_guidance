import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu


class TrafficLightPipeline(trr_vu.Pipeline):
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        #self.roi = np.array(
        self.img = None

    def _process_image(self, img, cam):
        self.img = img
        tl = np.array([150, 15])
        br = np.array([300, 100])
        w, h = br - tl
        tr = tl + [w, 0]
        bl = br - [w, 0]
        #vertices = np.array([tl, tr, br, bl], dtype=np.float32)
        self.roi_img = self.img[120:230,15:92]
        #mask = np.zeros_like(img)
        mask = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        cv2.rectangle(mask, tuple(tl), tuple(br), color=255, thickness=-1)#cv2.CV_FILLED)
        #cv2.fillPoly(mask, vertices, 255)
        self.masked_image = cv2.bitwise_and(img, mask)

    def draw_debug(self, cam, img_cam=None):
        if self.img is None:
            return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        out_img = self.masked_image
        
        self.draw_timing(out_img)
        return out_img
