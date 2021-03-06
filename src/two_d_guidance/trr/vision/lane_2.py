import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trr_u
#import rospy # maybe not...
import pdb

class Contour2Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam, robot_name, ctr_img_min_area=50):
        trr_vu.Pipeline.__init__(self)
        self.cam = cam
        be_param = trr_vu.NamedBirdEyeParam(robot_name)
        self.set_roi((0, 0), (cam.w, cam.h))
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder(min_area=ctr_img_min_area)
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.lane_model = trr_u.LaneModel()
        self.display_mode = Contour2Pipeline.show_none
        self.img = None
        self.cnt_max_be = None
        
    def init_masks(self): # FIXME, compute that from be params
        y0, x1, x2, y3, y4 = 100, 346, 452, 100, 30
        self.be_mask_roi = np.array( [ [0,0], [0, y0], [x1, y4], [x2, y4], [self.cam.w, y3],  [self.cam.w, 0] ] )
        self.be_mask_noroi = self.be_mask_roi + self.tl
        y0, x1, y1, x2, y3 = 350, 150, self.cam.h-20, 600, 350
        self.car_mask = np.array( [ [0, self.cam.h], [0, y0], [x1, y1], [x2,y1], [self.cam.w, y3],  [self.cam.w, self.cam.h] ] )
        self.car_mask_roi = self.car_mask - self.tl

    def set_roi(self, tl, br):
        print('roi: {} {}'.format(tl, br))
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        self.init_masks()
        
    def _process_image(self, img, cam, stamp):
        self.img = img
        self.img_roi = img[self.roi]
        self.thresholder.process_bgr(self.img_roi, birdeye_mode=False)

        ### masks...
        cv2.fillPoly(self.thresholder.threshold, [self.be_mask_roi, self.car_mask_roi], color=0)
        
        self.contour_finder.process(self.thresholder.threshold)
        self._process_all_contours(cam, stamp)
            
    def _process_all_contours(self, cam, stamp):
        ''' fit all valid contours '''
        self.cnts_be, self.cnts_lfp = [], []
        if self.contour_finder.valid_cnts is None: return

        for i, c in enumerate(self.contour_finder.valid_cnts):
            cnt_imp = cam.undistort_points((c+ self.tl).astype(np.float32))
            cnt_be = self.bird_eye.points_imp_to_be(cnt_imp)
            self.cnts_be.append(cnt_be)
            cnt_lfp = self.bird_eye.points_imp_to_blf(cnt_imp)
            cnt_lfp[cnt_lfp[:, 0, 0] < 0.01] = 20
            np.clip(cnt_lfp[:, 0, 0], None, 3., out=cnt_lfp[:, 0, 0])
            self.cnts_lfp.append(cnt_lfp)
        self.cnts_lfp = np.array(self.cnts_lfp)

        if len(self.cnts_lfp) > 0:
            self.lane_model.fit(self.cnts_lfp)
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
        self.lane_model.stamp = stamp

        
    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)
    
    def draw_debug_bgr(self, cam, img_cam=None, border_color=128):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Contour2Pipeline.show_input:
            debug_img = self.img
            # roi
            cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
            # masks
            #be_corners_img = self.bird_eye.param.corners_be_img.reshape((1, -1, 2)).astype(np.int)
            #cv2.polylines(debug_img, be_corners_img, isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(debug_img, [self.be_mask_noroi], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.polylines(debug_img, [self.car_mask], isClosed=True, color=(0, 255, 255), thickness=2)
        
        elif self.display_mode == Contour2Pipeline.show_thresh:
            roi_img =  cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == Contour2Pipeline.show_contour:
            roi_img = self._draw_contour(cam)
        elif self.display_mode == Contour2Pipeline.show_be:
            debug_img = self._draw_be(cam)

        if self.display_mode not in [Contour2Pipeline.show_input, Contour2Pipeline.show_be] :
            debug_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            try:
                debug_img[self.roi] = roi_img
            except UnboundLocalError:  # race condition: dispay mode has been changed while we were drawing
                pass
        if self.display_mode in [Contour2Pipeline.show_contour, Contour2Pipeline.show_input]:
            if self.lane_model.is_valid():
                self.lane_model.draw_on_cam_img(debug_img, cam, l0=self.lane_model.x_min, l1=self.lane_model.x_max, color=(0,128,255))
                #self.lane_model.draw_on_cam_img(debug_img, cam, l0=0.3, l1=1.2)
    
        self._draw_HUD(debug_img)
        # we return a BGR image
        return debug_img

    def _draw_HUD(self, debug_img):
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        h1, c1, dy = 1., (220, 130, 120), 30
        cv2.putText(debug_img, 'Lane#2', (20, 40), f, h, c, w)
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
        roi_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        self.contour_finder.draw2(roi_img, self.contour_finder.valid_cnts, self.lane_model.inliers_mask)
        #self.contour_finder.draw(roi_img, draw_all=True)
        cv2.fillPoly(roi_img, [self.be_mask_roi, self.car_mask_roi], color=mask_color)
        return roi_img
        
    def _draw_be(self, cam):
        try:
            undistorted_img = cam.undistort_img_map(self.img)
            #unwarped_img = self.bird_eye.process(undistorted_img)
            unwarped_img = self.bird_eye.unwarp_map(undistorted_img)
            debug_img = self.bird_eye.draw_debug(cam, unwarped_img, self.lane_model, self.cnts_be)
        except AttributeError:
            debug_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        return debug_img

