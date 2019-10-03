import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trr_u
#import rospy # maybe not...
import pdb

class Contour3Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam, robot_name):
        trr_vu.Pipeline.__init__(self)
        be_param = trr_vu.NamedBirdEyeParam(robot_name)
        self.cam = cam
        self.set_roi((0, 0), (cam.w, cam.h))
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder(min_area=500)
        self.lane_model = trr_u.LaneModel()
        self.display_mode = Contour3Pipeline.show_none
        self.img = None
        
    def set_roi(self, tl, br): pass

    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_unwarped = self.bird_eye.undist_unwarp_map(img, cam)
        self.thresholder.process_bgr(self.img_unwarped, True)
        ### masks...
        cv2.fillPoly(self.thresholder.threshold, [self.bird_eye.mask_unwraped], color=0)
        
        self.contour_finder.process(self.thresholder.threshold)
        self.cnts_be = self.contour_finder.valid_cnts if self.contour_finder.valid_cnts is not None else []
        self.cnts_lfp = [self.bird_eye.unwarped_to_fp(cam, _c_be)[:,:2].reshape((-1,1,2)).astype(np.float32) for _c_be in self.cnts_be]
        self.cnts_lfp = np.array(self.cnts_lfp)
        if len(self.cnts_lfp) > 0:
            self.lane_model.fit(self.cnts_lfp)
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_invalid()
        self.lane_model.stamp = stamp



    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)
    
    def draw_debug_bgr(self, cam, img_cam=None, border_color=128):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Contour3Pipeline.show_input:
            debug_img = self.img
            # roi
            #cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
            # masks
            #be_corners_img = self.bird_eye.param.corners_be_img.reshape((1, -1, 2)).astype(np.int)
            #cv2.polylines(debug_img, be_corners_img, isClosed=True, color=(0, 0, 255), thickness=2)
            #cv2.polylines(debug_img, [self.be_mask_noroi], isClosed=True, color=(0, 255, 255), thickness=2)
            #cv2.polylines(debug_img, [self.car_mask], isClosed=True, color=(0, 255, 255), thickness=2)
        
        elif self.display_mode == Contour3Pipeline.show_thresh:
            be_img =  cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            debug_img = trr_vu.change_canvas(be_img, cam.h, cam.w)
            
        elif self.display_mode == Contour3Pipeline.show_contour:
            debug_img = self._draw_contour(cam)
            
        elif self.display_mode == Contour3Pipeline.show_be:
            debug_img = self._draw_be(cam)

        if self.display_mode in [Contour3Pipeline.show_input]:
            if self.lane_model.is_valid():
                self.lane_model.draw_on_cam_img(debug_img, cam, l0=self.lane_model.x_min, l1=self.lane_model.x_max, color=(0,128,255))
                #self.lane_model.draw_on_cam_img(debug_img, cam, l0=0.3, l1=1.2)
    
        self._draw_HUD(debug_img)
        # we return a BGR image
        return debug_img

    def _draw_HUD(self, debug_img):
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        h1, c1, dy = 1., (220, 130, 120), 30
        cv2.putText(debug_img, 'Lane#3', (20, 40), f, h, c, w)
        self.draw_timing(debug_img, x0=360, y0=40)
        try:
            nb_valid_contours = len(self.contour_finder.valid_cnts)
        except TypeError:
            #rospy.loginfo_throttle(1., "Lane2: no valid contour") # print every second
            nb_valid_contours = 0
        y0 = 120
        cv2.putText(debug_img, 'valid contours: {}'.format(nb_valid_contours), (20, y0), f, h1, c1, w)
        cv2.putText(debug_img, 'model: {} valid'.format('' if self.lane_model.is_valid() else 'not'), (20, y0+dy), f, h1, c1, w)

    
    def _draw_contour(self, cam, mask_color=(150, 150, 120)):
        debug_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        self.contour_finder.draw2(debug_img, self.contour_finder.valid_cnts, self.lane_model.inliers_mask)
        #cv2.fillPoly(roi_img, [self.be_mask_roi, self.car_mask_roi], color=mask_color)
        if self.lane_model.is_valid():
            self.bird_eye.draw_lane(cam, debug_img, self.lane_model, color=(128,128,255))
        return debug_img
        
    def _draw_be(self, cam):
        try:
            debug_img = self.bird_eye.draw_debug(cam, self.img_unwarped, self.lane_model, self.cnts_be)
        except AttributeError:
            debug_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        return debug_img

