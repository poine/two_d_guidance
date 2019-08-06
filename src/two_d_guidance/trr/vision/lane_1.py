import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_utils as trru

class Contour1Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam):
        trr_vu.Pipeline.__init__(self)
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder()
        self.floor_plane_injector = trr_vu.FloorPlaneInjector()
        self.lane_model = trru.LaneModel()
        self.display_mode = Contour1Pipeline.show_contour
        self.img = None
        
    def _process_image(self, img, cam):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.thresholder.process(self.img_gray)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_max is not None:
            self.floor_plane_injector.compute(self.contour_finder.cnt_max, cam)
            self.lane_model.fit(self.floor_plane_injector.contour_floor_plane_blf[:,:2])
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
        
    def draw_debug(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3))
        if self.display_mode == Contour1Pipeline.show_thresh:
            out_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR) # cv2.COLOR_GRAY2BGR ??
        else:
            out_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.draw_timing(out_img)
        self.contour_finder.draw(out_img)
        if self.lane_model.is_valid():
            self.lane_model.draw_on_cam_img(out_img, cam)
        return out_img
        
