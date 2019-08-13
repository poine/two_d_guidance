import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trru


class StartFinishDetectPipeline(trr_vu.Pipeline):
    show_none, show_input, show_red_mask, show_green_mask, show_contour, show_be = range(6)
    
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.ss_dtc = trr_vu.StartFinishDetector()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.img_bgr = None
        self.start_ctr_lfp, self.finish_ctr_lfp = None, None
        self.ctrs_lfp = [None, None]
        self.dists_to_ctrs = [0, 0]
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

    def _localise_contour(self, ctr_idx, cam):
        if self.ss_dtc.sees_ctr(ctr_idx):
            _ctr = self.ss_dtc.get_contour(ctr_idx) + self.roi_tl
            #_ctr2 = self.ss_dtc.red_ccf.get_contour() + self.roi_tl
            #print _ctr[0], _ctr2[0]
            #print('loc finish_ctr {} {}'.format(ctr_idx, _ctr))
            _ctr_imp = cam.undistort_points(_ctr.astype(np.float32))
            _ctr_be = self.bird_eye.points_imp_to_be(_ctr_imp)
            self.ctrs_lfp[ctr_idx] = self.bird_eye.unwarped_to_fp(cam, _ctr_be)
            #print('loc lfp {} {}'.format(ctr_idx, self.ctrs_lfp[ctr_idx][0]))  # good
            #print type(self.ctrs_lfp[ctr_idx])
            M = cv2.moments(self.ctrs_lfp[ctr_idx]); m00=M['m00']
            cx = M['m10']/m00 if abs(m00) > 1e-9 else 0
            cy = M['m01']/m00 if abs(m00) > 1e-9 else 0
            #print('loc {} cx {}  cy {}'.format(ctr_idx, cx, cy)) # bad!!!!
            # centroid in local floor plane
            c_lfp = self.bird_eye.unwarped_to_fp(cam, np.array([[cx, cy], [cx, cy]]))[0]
            #print c_lfp
            self.dists_to_ctrs[ctr_idx] = np.linalg.norm(c_lfp)
        else:
            self.ctrs_lfp[ctr_idx] = None
            self.dists_to_ctrs[ctr_idx]  = 0
            
    def _process_image(self, img_bgr, cam):
        self.img_bgr = img_bgr
        self.roi_img_bgr = self.img_bgr[self.roi]
        self.ss_dtc.process_image(self.roi_img_bgr)
        for i in range(self.ss_dtc.CTR_NB):
            self._localise_contour(i, cam)
        
        self.dist_to_start = self.dists_to_ctrs[self.ss_dtc.CTR_START]
        self.dist_to_finish = self.dists_to_ctrs[self.ss_dtc.CTR_FINISH]
        #print('loc d:{}'.format(self.dist_to_finish))
        self.start_ctr_lfp = self.ctrs_lfp[self.ss_dtc.CTR_START]
        self.finish_ctr_lfp = self.ctrs_lfp[self.ss_dtc.CTR_FINISH]

        if self.ss_dtc.sees_finish():
            finish_ctr = self.ss_dtc.red_ccf.get_contour() + self.roi_tl
            #print('truth finish_ctr {}'.format(finish_ctr))
            finish_ctr_imp = cam.undistort_points(finish_ctr.astype(np.float32))
            self.finish_ctr_be = self.bird_eye.points_imp_to_be(finish_ctr_imp)
            self.finish_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.finish_ctr_be)
            #print('truth lfp {}'.format(self.finish_ctr_lfp[0]))
            #print type(self.finish_ctr_lfp)
            M = cv2.moments(self.finish_ctr_be); m00=M['m00']
            cx = M['m10']/m00 if abs(m00) > 1e-9 else 0
            cy = M['m01']/m00 if abs(m00) > 1e-9 else 0
            #print('truth cx {}  cy {}'.format(cx, cy))
            #print cx, cy
            c_lfp = self.bird_eye.unwarped_to_fp(cam, np.array([[cx, cy], [cx, cy]]))[0]
            self.dist_to_finish = np.linalg.norm(c_lfp)
            #print('truth df:{}'.format(self.dist_to_finish))
        else:
            self.dist_to_finish = float('inf')

        if self.ss_dtc.sees_start():
            start_ctr = self.ss_dtc.green_ccf.get_contour() + self.roi_tl
            start_ctr_imp = cam.undistort_points(start_ctr.astype(np.float32))
            self.start_ctr_be = self.bird_eye.points_imp_to_be(start_ctr_imp)
            self.start_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.start_ctr_be)
            #print('truth lfp {}'.format(self.finish_ctr_lfp[0]))
            #print type(self.finish_ctr_lfp)
            M = cv2.moments(self.start_ctr_be); m00=M['m00']
            cx = M['m10']/m00 if abs(m00) > 1e-9 else 0
            cy = M['m01']/m00 if abs(m00) > 1e-9 else 0
            #print('truth cx {}  cy {}'.format(cx, cy))
            #print cx, cy
            c_lfp = self.bird_eye.unwarped_to_fp(cam, np.array([[cx, cy], [cx, cy]]))[0]
            self.dist_to_start = np.linalg.norm(c_lfp)
            #print('truth ds:{}'.format(self.dist_to_start))
        else:
            self.dist_to_start = float('inf')

            #print(' ')
      
            

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
            roi_img_h, roi_img_w, _ = roi_img.shape     # FIXME: synch issue
            if self.roi_h == roi_img_h and self.roi_w == roi_img_w:
                out_img[self.roi] = roi_img
        if self.show_hud: self.draw_hud(out_img, cam)
        # we return a RGB8 image
        return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)


    def draw_hud(self, out_img, cam, y0=20, font_color=(0,255,255)):
        self.draw_timing(out_img, x0=350)   
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, font_color, 2
        tg, tr = 'no', 'no'
        if self.ss_dtc.green_ccf.has_contour():
            tg = 'area: {:.1f} dist: {:.2f}'.format(self.ss_dtc.green_ccf.get_contour_area(), self.dist_to_start)
        if self.ss_dtc.red_ccf.has_contour():
            tr = 'area: {:.1f} dist: {:.2f}'.format(self.ss_dtc.red_ccf.get_contour_area(), self.dist_to_finish)
        cv2.putText(out_img, 'StartFinish', (y0, 40), f, h, c, w)
        cv2.putText(out_img, 'start:  {}'.format(tg), (y0, cam.h-70), f, h, c, w)
        cv2.putText(out_img, 'finish: {}'.format(tr), (y0, cam.h-20), f, h, c, w)
        
