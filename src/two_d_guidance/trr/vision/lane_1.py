import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trru

import pdb
# TODO: use multiple contours rather than larger one?
class Contour1Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_thresh, show_contour, show_be = range(5)
    def __init__(self, cam):
        trr_vu.Pipeline.__init__(self)
        self.thresholder = trr_vu.BinaryThresholder()
        self.contour_finder = trr_vu.ContourFinder(min_area=100)
        self.floor_plane_injector = trr_vu.FloorPlaneInjector()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam,  trr_vu.BirdEyeParam(x0=-0.3, dx=3., dy=3., w=480))
        self.lane_model = trru.LaneModel()
        self.display_mode = Contour1Pipeline.show_contour
        self.img = None

    def set_roi(self, tl, br):
        print('roi: {} {}'.format(tl, br))
        self.tl, self.br = tl, br
        self.roi_h, self.roi_w = self.br[1]-self.tl[1], self.br[0]-self.tl[0]
        self.roi = slice(self.tl[1], self.br[1]), slice(self.tl[0], self.br[0])
        
    def _process_image(self, img, cam):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.thresholder.process(self.img_gray)
        self.contour_finder.process(self.thresholder.threshold)
        if self.contour_finder.cnt_max is not None:
            self.floor_plane_injector.compute(self.contour_finder.cnt_max, cam)
            self.cnt_max_blf = self.floor_plane_injector.contour_floor_plane_blf[:,:2]
            x_min, x_max   = np.min(self.cnt_max_blf[:,0]), np.max(self.cnt_max_blf[:,0])
            y_min, y_max = np.min(self.cnt_max_blf[:,1]), np.max(self.cnt_max_blf[:,1])
            print('x in [{:.2f} {:.2f}] y in [{:.2f} {:.2f}]'.format(x_min, x_max, y_min, y_max))
            self.lane_model.fit(self.cnt_max_blf)
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)
        
    def draw_debug(self, cam, img_cam=None):
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam), cv2.COLOR_BGR2RGB)

    def draw_debug_bgr(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == Contour1Pipeline.show_input:
            debug_img = self.img
        elif self.display_mode == Contour1Pipeline.show_thresh:
            debug_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == Contour1Pipeline.show_contour:
            debug_img = cv2.cvtColor(self.thresholder.threshold, cv2.COLOR_GRAY2BGR)
            self.contour_finder.draw(debug_img, draw_all=True)
        elif self.display_mode == Contour1Pipeline.show_be:
            debug_img = self.draw_be_scene(cam)

        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        cv2.putText(debug_img, 'Lane #1', (20, 40), f, h, c, w)
        self.draw_timing(debug_img)
        
        if self.lane_model.is_valid() and self.display_mode not in [Contour1Pipeline.show_be]:
            x_min, x_max = self.lane_model.x_min, self.lane_model.x_max
            #x_min, x_max = 0.15, 12.
            self.lane_model.draw_on_cam_img(debug_img, cam, l0=x_min, l1=x_max, color=(128, 128, 0))
        return debug_img
            
    def draw_be_scene(self, cam):
        be_img = np.zeros((self.bird_eye.h, self.bird_eye.w, 3), dtype=np.uint8)
        
        # draw blf frame axis
        pts_blf = np.array([[0, 0, 0], [1, 0, 0],
                            [0, 0, 0], [0, 1, 0]], dtype=np.float32)
        pts_img = self.bird_eye.lfp_to_unwarped(cam, pts_blf)
        color = (128, 0, 0)
        for i in range(len(pts_img)-1):
            cv2.line(be_img, tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)), color, 4)

        if self.lane_model.is_valid():
            self.bird_eye.draw_lane(cam, be_img, self.lane_model, self.lane_model.x_min, self.lane_model.x_max)
        

            
        #debug_img = self.bird_eye.draw_debug(cam, img=debug_img, lane_model=self.lane_model)
        debug_img = trr_vu.change_canvas(be_img, cam.h, cam.w, border_color=(128, 128, 128))
        
        
        return debug_img
