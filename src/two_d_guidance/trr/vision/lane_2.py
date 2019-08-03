import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_utils as trr_u

class Contour2Pipeline(trr_vu.Pipeline):
    show_be, show_thresh, show_contour = range(3)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.lane_model = trr_u.LaneModel()
        self.display_mode = Contour2Pipeline.show_be
        self.img = None
        
    def _process_image(self, img, cam):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        self.thresholder.process(self.img_gray)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_max is not None:
            cnt_max_imp = cam.undistort_points(self.contour_finder.cnt_max.astype(np.float32))
            #cnt_max_imp = cam.undistort_points2(self.contour_finder.cnt_max.astype(np.float32)) # TODO
            self.cnt_max_be = self.bird_eye.points_imp_to_be(cnt_max_imp)
            self.cnt_max_lfp = self.bird_eye.unwarped_to_fp(cam, self.cnt_max_be)
            self.lane_model.fit(self.cnt_max_lfp[:,:2])
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
            
        
    def draw_debug(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3))
        if self.display_mode == Contour2Pipeline.show_be:
            img = self.bird_eye.draw_debug(cam, None, self.lane_model, self.cnt_max_be.astype(np.int32))
            self.draw_timing(img)
            return img
        elif self.display_mode == Contour2Pipeline.show_thresh:
            img =  cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.draw_timing(img)
            return img
        elif self.display_mode == Contour2Pipeline.show_contour:
            img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.contour_finder.draw(img)
            if self.lane_model.is_valid(): self.lane_model.draw_on_cam_img(img, cam)
            self.draw_timing(img)
            return img
