import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_utils as trr_u

import pdb

class Contour2Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.set_roi((0, 130), (cam.w, cam.h))
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.lane_model = trr_u.LaneModel()
        self.display_mode = Contour2Pipeline.show_be
        self.img = None
        self.cnt_max_be = None

    def set_roi(self, tl, br):
        print('roi: {}'.format(tl, br))
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        
    def _process_image(self, img, cam):
        self.img = img
        self.img_roi = img[self.roi]
        self.img_gray = cv2.cvtColor(self.img_roi, cv2.COLOR_BGR2GRAY )
        self.thresholder.process(self.img_gray)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_max is not None:
            cnt_max_noroi = self.contour_finder.cnt_max + self.tl
            cnt_max_imp = cam.undistort_points(cnt_max_noroi.astype(np.float32))
            #cnt_max_imp = cam.undistort_points2(self.contour_finder.cnt_max.astype(np.float32)) # TODO
            self.cnt_max_be = self.bird_eye.points_imp_to_be(cnt_max_imp)
            self.cnt_max_lfp = self.bird_eye.unwarped_to_fp(cam, self.cnt_max_be)
            self.lane_model.fit(self.cnt_max_lfp[:,:2])
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
            
        
    def draw_debug(self, cam, img_cam=None, border_color=128):
        if self.img is None: return np.zeros((cam.h, cam.w, 3))
        if self.display_mode == Contour2Pipeline.show_input:
            debug_img = self.img
            cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == Contour2Pipeline.show_thresh:
            roi_img =  cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == Contour2Pipeline.show_contour:
            roi_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.contour_finder.draw(roi_img)
        elif self.display_mode == Contour2Pipeline.show_be:
            debug_img = self.bird_eye.draw_debug(cam, None, self.lane_model, self.cnt_max_be.astype(np.int32))
            #debug_img = self.bird_eye.draw_debug(cam, None, self.lane_model, None)
        if self.display_mode not in [Contour2Pipeline.show_input, Contour2Pipeline.show_be] :
            debug_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            debug_img[self.roi] = roi_img
        if self.display_mode in [Contour2Pipeline.show_contour]:
            if self.lane_model.is_valid(): self.lane_model.draw_on_cam_img(debug_img, cam)
            
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        cv2.putText(debug_img, 'Lane detection', (20, 40), f, h, c, w)
        self.draw_timing(debug_img, y0=90)
        # we return a RGB image
        return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        #return debug_img