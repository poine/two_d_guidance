import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trru

import pdb

class StartFinishDetectPipeline(trr_vu.Pipeline):
    show_none, show_input, show_red_mask, show_green_mask, show_contour, show_be = range(6)
    
    def __init__(self, cam, robot_name):
        trr_vu.Pipeline.__init__(self)
        self.sfd = trr_vu.StartFinishDetector()

        self.dist_to_start, self.dist_to_finish = np.float('inf'), np.float('inf')

        self.bird_eye = trr_vu.BirdEyeTransformer(cam, trr_vu.NamedBirdEyeParam(robot_name))
        self.img_bgr = None

        self.set_debug_display(StartFinishDetectPipeline.show_none, False)
        self.set_roi((0, 110), (cam.w, cam.h))

    def set_debug_display(self, display_mode, show_hud):
        self.display_mode, self.show_hud = display_mode, show_hud

    def set_green_mask_params(self, hc, hs, smin, smax, vmin, vmax, gray_thr):
        self.sfd.set_hsv_range(self.sfd.CTR_START, trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax))

    def set_red_mask_params(self, hc, hs, smin, smax, vmin, vmax, gray_thr):
        #param_txt = '{} {} / {} {} / {} {} / {}'.format(hc, hs, smin, smax, vmin, vmax, gray_thr)
        #print('StartFinishDetectPipeline::set_red_mask_params: {}'.format(param_txt))
        self.sfd.set_hsv_range(self.sfd.CTR_FINISH, trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax))
        

    def get_red_mask_params(self):
        return self.sfd.red_ccf.get_hsv_range()

        
    def set_roi(self, tl, br):
        print('start_finish::set roi: {} {}'.format(tl, br))
        self.roi_tl, self.roi_br = tl, br
        self.roi_h, self.roi_w = br[1]-tl[1], br[0]-tl[0]
        self.roi = slice(tl[1], br[1]), slice(tl[0], br[0])

            
    def _process_image(self, img_bgr, cam):
        self.img_bgr = img_bgr
        self.roi_img_bgr = self.img_bgr[self.roi]
        self.sfd.process_image(self.roi_img_bgr)

        if self.sfd.sees_start():
            center_img = np.add(self.sfd.get_center_coords(self.sfd.CTR_START), self.roi_tl)
            center_imp = cam.undistort_points(center_img.reshape(1,-1,2))
            center_lfp = self.bird_eye.points_imp_to_blf(center_imp)
            self.dist_to_start = np.linalg.norm(center_lfp)
        else: self.dist_to_start = np.float('inf')

        if self.sfd.sees_finish():
            center_img = np.add(self.sfd.get_center_coords(self.sfd.CTR_FINISH), self.roi_tl)
            center_imp = cam.undistort_points(center_img.reshape(1,-1,2))
            center_lfp = self.bird_eye.points_imp_to_blf(center_imp)
            self.dist_to_finish = np.linalg.norm(center_lfp)
        else: self.dist_to_finish = np.float('inf')
            
            
    def draw_debug(self, cam, img_cam=None, border_color=128):
        # we return a RGB8 image
        return cv2.cvtColor(self.draw_debug_bgr(cam, img_cam, border_color), cv2.COLOR_BGR2RGB)

    def draw_debug_bgr(self, cam, img_cam=None, border_color=128):
        if self.img_bgr is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == StartFinishDetectPipeline.show_input:
            out_img = self.img_bgr
            cv2.rectangle(out_img, tuple(self.roi_tl), tuple(self.roi_br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == StartFinishDetectPipeline.show_red_mask:
            roi_img = cv2.cvtColor(self.sfd.masks[self.sfd.CTR_FINISH], cv2.COLOR_GRAY2BGR)
        elif self.display_mode == StartFinishDetectPipeline.show_green_mask:
            roi_img = cv2.cvtColor(self.sfd.masks[self.sfd.CTR_START], cv2.COLOR_GRAY2BGR)
        elif self.display_mode == StartFinishDetectPipeline.show_contour:
            roi_img = self.sfd.draw(None)
        # elif self.display_mode == StartFinishDetectPipeline.show_be:
        #     roi_img = self.sfd.draw(self.roi_img_bgr)
        if self.display_mode != StartFinishDetectPipeline.show_input:
            out_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            try:
                out_img[self.roi] = roi_img
            except TypeError:
                print('type error')
        if self.show_hud: self.draw_hud(out_img, cam)
        # we return a BGR8 image
        return out_img


    def draw_hud(self, out_img, cam, y0=20, font_color=(0,255,255)):
        self.draw_timing(out_img, x0=350)   
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, font_color, 2
        tg, tr = 'no', 'no'
        if self.sfd.sees_start():  tg = 'area: {:.1f} dist: {:.2f}'.format(0, self.dist_to_start)
        if self.sfd.sees_finish(): tr = 'area: {:.1f} dist: {:.2f}'.format(0, self.dist_to_finish)
        cv2.putText(out_img, 'StartFinish', (y0, 40), f, h, c, w)
        cv2.putText(out_img, 'start:  {}'.format(tg), (y0, cam.h-70), f, h, c, w)
        cv2.putText(out_img, 'finish: {}'.format(tr), (y0, cam.h-20), f, h, c, w)
        
