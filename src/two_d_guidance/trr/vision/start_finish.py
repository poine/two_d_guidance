import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_utils as trru


class StartFinishDetectPipeline(trr_vu.Pipeline):
    show_none, show_input, show_red_mask, show_green_mask, show_contour, show_be = range(6)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.ss_dtc = trr_vu.StartFinishDetector()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.img_bgr = None
        self.start_ctr_lfp, self.finish_ctr_lfp = None, None
        self.set_debug_display(StartFinishDetectPipeline.show_none, False)
        self.set_roi((0, 0), (cam.w, cam.h))

    def set_debug_display(self, display_mode, show_hud):
        self.display_mode, self.show_hud = display_mode, show_hud

    def set_green_mask_params(self, hc, hs, smin, smax, vmin, vmax, gray_thr):
        self.ss_dtc.green_ccf.set_hsv_range(trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax))
        self.ss_dtc.green_ccf.set_gray_threshold(gray_thr)

    def set_red_mask_params(self, hc, hs, smin, smax, vmin, vmax, gray_thr):
        self.ss_dtc.red_ccf.set_hsv_range(trr_vu.hsv_range(hc, hs, smin, smax, vmin, vmax))
        self.ss_dtc.red_ccf.set_gray_threshold(gray_thr)

    def set_roi(self, tl, br):
        print('roi: {} {}'.format(tl, br))
        self.roi_tl, self.roi_br = tl, br
        self.roi_h, self.roi_w = br[1]-tl[1], br[0]-tl[0]
        self.roi = slice(tl[1], br[1]), slice(tl[0], br[0])
        
    def _process_image(self, img_bgr, cam):
        self.img_bgr = img_bgr
        self.roi_img_bgr = self.img_bgr[self.roi]
        self.ss_dtc.process_image(self.roi_img_bgr)
        if self.ss_dtc.sees_start():
            start_ctr = self.ss_dtc.green_ccf.get_contour() + self.roi_tl
            start_ctr_imp = cam.undistort_points(start_ctr.astype(np.float32))
            self.start_ctr_be = self.bird_eye.points_imp_to_be(start_ctr_imp)
            self.start_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.start_ctr_be)
            self.dist_to_start = 0
        else:
            self.start_ctr_lfp = None

        if self.ss_dtc.sees_finish():
            finish_ctr = self.ss_dtc.red_ccf.get_contour() + self.roi_tl
            finish_ctr_imp = cam.undistort_points(finish_ctr.astype(np.float32))
            self.finish_ctr_be = self.bird_eye.points_imp_to_be(finish_ctr_imp)
            self.finish_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.finish_ctr_be)
            # centroid in bird eye image
            M = cv2.moments(self.finish_ctr_be); m00=M['m00']
            cx = M['m10']/m00 if abs(m00) > 1e-9 else 0
            cy = M['m01']/m00 if abs(m00) > 1e-9 else 0
            #print cx, cy
            # centroid in local floor plane
            c_lfp = self.bird_eye.unwarped_to_fp(cam, np.array([[cx, cy], [cx, cy]]))[0]
            #print c_lfp
            self.dist_to_finish = np.linalg.norm(c_lfp)
        else:
            self.red_ctr_lfp = None
            

    def draw_debug(self, cam, img_cam=None, border_color=128):
        if self.img_bgr is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == StartFinishDetectPipeline.show_input:
            out_img = self.img_bgr
            cv2.rectangle(out_img, tuple(self.roi_tl), tuple(self.roi_br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == StartFinishDetectPipeline.show_red_mask:
            roi_img = cv2.cvtColor(self.ss_dtc.red_ccf.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == StartFinishDetectPipeline.show_green_mask:
            roi_img = cv2.cvtColor(self.ss_dtc.green_ccf.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == StartFinishDetectPipeline.show_contour:
            roi_img = self.ss_dtc.draw(self.roi_img_bgr)
        elif self.display_mode == StartFinishDetectPipeline.show_be:
            roi_img = self.ss_dtc.draw(self.roi_img_bgr)
        if self.display_mode != StartFinishDetectPipeline.show_input:
            out_img = np.full((cam.h, cam.w, 3), border_color, dtype=np.uint8)
            out_img[self.roi] = roi_img
        if self.show_hud: self.draw_hud(out_img, cam)
        # we return a RGB8 image
        return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)


    def draw_hud(self, out_img, cam):
        self.draw_timing(out_img, x0=300)   
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2
        tg, tr = 'no', 'no'
        if self.ss_dtc.green_ccf.has_contour():
            tg = 'area: {:.1f} dist: {:.2f}'.format(self.ss_dtc.green_ccf.get_contour_area(), self.dist_to_start)
        if self.ss_dtc.red_ccf.has_contour():
            tr = 'area: {:.1f} dist: {:.2f}'.format(self.ss_dtc.red_ccf.get_contour_area(), self.dist_to_finish)
        cv2.putText(out_img, 'start: {}'.format(tg), (20, cam.h-70), f, h, c, w)
        cv2.putText(out_img, 'finish: {}'.format(tr), (20, cam.h-20), f, h, c, w)
        
