import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trr_u
#import rospy # maybe not...
import pdb

class Contour2Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam(), use_single_contour=False, ctr_img_min_area=200):
        trr_vu.Pipeline.__init__(self)
        self.use_single_contour = use_single_contour
        self.set_roi((0, 0), (cam.w, cam.h))
        #self.set_roi((0, 70), (cam.w, cam.h))
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder(min_area=ctr_img_min_area)
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.lane_model = trr_u.LaneModel()
        self.display_mode = Contour2Pipeline.show_none
        self.img = None
        self.cnt_max_be = None
        self.use_single_contour = use_single_contour

    def set_roi(self, tl, br):
        print('roi: {} {}'.format(tl, br))
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        
    def _process_image(self, img, cam):
        self.img = img
        self.img_roi = img[self.roi]
        self.img_gray = cv2.cvtColor(self.img_roi, cv2.COLOR_BGR2GRAY )
        self.thresholder.process(self.img_gray)
        self.contour_finder.process(self.thresholder.threshold)
        if self.use_single_contour:
            self._process_max_area_contour(cam)
        else:
            self._process_all_contours(cam)
            
    def _process_max_area_contour(self, cam):
        ''' fit contour with max area (in img plan) '''
        if self.contour_finder.has_contour():
            cnt_max_noroi = self.contour_finder.get_contour() + self.tl
            cnt_max_imp = cam.undistort_points(cnt_max_noroi.astype(np.float32))
            self.cnt_max_be = self.bird_eye.points_imp_to_be(cnt_max_imp)
            self.cnt_max_lfp = self.bird_eye.unwarped_to_fp(cam, self.cnt_max_be)
            self.lane_model.fit(self.cnt_max_lfp[:,:2])
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)

    def _process_all_contours(self, cam):
        ''' fit all valid contours '''
        self._compute_contours_lfp(cam)
        # if 0: # sorted by img area , broken
        #     sorted_cnts_lfp = self.cnts_lfp[self.contour_finder.valib_cnt_areas_order]
        #     nb_ctr = min(len(self.cnts_lfp), 3)
        #     sum_ctr_lfp = np.concatenate(sorted_cnts_lfp[nb_ctr::-1], axis=0)
        # else: #  all
        if len(self.cnts_lfp) > 0:
            sum_ctr_lfp = np.concatenate(self.cnts_lfp, axis=0)
            self.lane_model.fit(sum_ctr_lfp.squeeze())
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
            

            
    def _compute_contours_lfp(self, cam):
        self.cnts_be, self.cnts_lfp = [], []
        if self.contour_finder.valid_cnts is None: return

        if 0:
            for i, c in enumerate(self.contour_finder.valid_cnts):
                #print('ctr {}'.format(i))
                # Compute contour vertices in optical plan (undistort)
                cnt_imp = cam.undistort_points((c+ self.tl).astype(np.float32))
                #trr_u.print_extends(cnt_imp.squeeze(), ' undis')
                # Compute contour vertices in bird eye coordinates
                cnt_be = self.bird_eye.points_imp_to_be(cnt_imp)
                #trr_u.print_extends(cnt_be.squeeze(),  ' be   ', full=True)
                # Filter out points that are outside bird eye area
                cnt_be1 = []
                #print('  shape undist {}'.format(cnt_be.shape))
                for p in cnt_be.squeeze():
                    if p[0] >= 0 and p[0] < self.bird_eye.w and p[1]>=0 and p[1]<self.bird_eye.h:
                        cnt_be1.append(p)
                cnt_be1 = np.array(cnt_be1).reshape(-1, 1, 2)
                #print('  shape undist2 {}'.format(cnt_be1.shape))
                min_vertices = 5
                if cnt_be1.shape[0] > min_vertices: # we have not removed all the contour
                    #trr_u.print_extends(cnt_be1.squeeze(),  ' be1   ')
                    self.cnts_be.append(cnt_be1)
                    try:
                        cnt_lfp = self.bird_eye.unwarped_to_fp(cam, cnt_be1)
                    except IndexError:
                        pdb.set_trace()
                        #trr_u.print_extends(cnt_lfp.squeeze(),  ' lfp   ')
                        self.cnts_lfp.append(cnt_lfp)
                #else:
                #    self.cnts_be.append(c_be1)  # empty to keep sorting happy
                #    self.cnts_lfp.append(c_be1)
                
        
        for i, c in enumerate(self.contour_finder.valid_cnts):
            cnt_imp = cam.undistort_points((c+ self.tl).astype(np.float32))
            #cnt_be = self.bird_eye.points_imp_to_be(cnt_imp)
            #self.cnts_be.append(cnt_be)
            cnt_lfp = self.bird_eye.points_imp_to_blf(cnt_imp)
            self.cnts_lfp.append(cnt_lfp)
        self.cnts_lfp = np.array(self.cnts_lfp)
        
            
    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)
    
    def draw_debug_bgr(self, cam, img_cam=None, border_color=128):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Contour2Pipeline.show_input:
            debug_img = self.img
            cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == Contour2Pipeline.show_thresh:
            roi_img =  cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == Contour2Pipeline.show_contour:
            roi_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.contour_finder.draw(roi_img, draw_all=True)
        elif self.display_mode == Contour2Pipeline.show_be:
            debug_img = self._draw_be(cam)

        if self.display_mode not in [Contour2Pipeline.show_input, Contour2Pipeline.show_be] :
            debug_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            debug_img[self.roi] = roi_img
        if self.display_mode in [Contour2Pipeline.show_contour]:
            if self.lane_model.is_valid():
                self.lane_model.draw_on_cam_img(debug_img, cam, l0=self.lane_model.x_min, l1=self.lane_model.x_max)
                #self.lane_model.draw_on_cam_img(debug_img, cam, l0=0.3, l1=1.2)
            
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        h1, c1, dy = 1., (18, 200, 5), 30
        cv2.putText(debug_img, 'Lane#2', (20, 40), f, h, c, w)
        self.draw_timing(debug_img, x0=360, y0=40)
        try:
            nb_valid_contours = len(self.contour_finder.valid_cnts)
        except TypeError:
            #rospy.loginfo_throttle(1., "Lane2: no valid contour") # print every second
            nb_valid_contours = 0
        cv2.putText(debug_img, 'valid contours: {}'.format(nb_valid_contours), (20, 90), f, h1, c1, w)
        cv2.putText(debug_img, 'model: {} valid'.format('' if self.lane_model.is_valid() else 'not'), (20, 90+dy), f, h1, c1, w)
        # we return a BGR image
        return debug_img

    def _draw_be(self, cam):
        try:
            if self.use_single_contour:
                debug_img = self.bird_eye.draw_debug(cam, None, self.lane_model, [self.cnt_max_be])
            else:
                debug_img = self.bird_eye.draw_debug(cam, None, self.lane_model, self.cnts_be)
        except AttributeError:
            debug_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        return debug_img

