import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_utils as trru


class StartFinishDetectPipeline(trr_vu.Pipeline):
    show_none, show_be, show_red_mask, show_green_mask, show_contour = range(5)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.ss_dtc = trr_vu.StartFinishDetector()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.img = None
        self.green_ctr_lfp = None
        self.red_ctr_lfp = None
        self.display_mode = StartFinishDetectPipeline.show_none
 
    def _process_image(self, img, cam):
        self.img = img
        self.ss_dtc.process_image(img)
        if self.ss_dtc.green_ccf.has_contour():
            green_ctr_imp = cam.undistort_points(self.ss_dtc.green_ccf.get_contour().astype(np.float32))
            self.green_ctr_be = self.bird_eye.points_imp_to_be(green_ctr_imp)
            self.green_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.green_ctr_be)
            #M = cv2.moments(self.green_ctr_be)
            #print M
        else:
            self.green_ctr_lfp = None

        if self.ss_dtc.red_ccf.has_contour():
            red_ctr_imp = cam.undistort_points(self.ss_dtc.red_ccf.get_contour().astype(np.float32))
            self.red_ctr_be = self.bird_eye.points_imp_to_be(red_ctr_imp)
            self.red_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.red_ctr_be)
            # centroid in bird eye image
            M = cv2.moments(self.red_ctr_be); m00=M['m00']
            cx = M['m10']/m00 if abs(m00) > 1e-9 else 0
            cy = M['m01']/m00 if abs(m00) > 1e-9 else 0
            #print cx, cy
            # centroid in local floor plane
            c_lfp = self.bird_eye.unwarped_to_fp(cam, np.array([[cx, cy], [cx, cy]]))[0]
            #print c_lfp
            self.dist_to_finish = np.linalg.norm(c_lfp)
        else:
            self.red_ctr_lfp = None
            

    def draw_debug(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        if self.display_mode == StartFinishDetectPipeline.show_be:
            out_img = self.ss_dtc.draw(self.img)
        elif self.display_mode == StartFinishDetectPipeline.show_red_mask:
            out_img = cv2.cvtColor(self.ss_dtc.red_ccf.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == StartFinishDetectPipeline.show_green_mask:
            out_img = cv2.cvtColor(self.ss_dtc.green_ccf.mask, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == StartFinishDetectPipeline.show_contour:
            out_img = self.ss_dtc.draw(self.img)
        self.draw_timing(out_img)   
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2
        #cds = [self.ss_dtc.contour_green, self.ss_dtc.contour_red]
        #tg, tr = ['no' if _cd.cnt_max is None else '{}'.format(_cd.cnt_max_area) for _cd in cds] 
        #tg = 'no' if self.ss_dtc.contour_green.cnt_max else 'area {:.1f}'.format(self.ss_dtc.contour_green.cnt_max_area)
        tg = 'no' if not self.ss_dtc.green_ccf.has_contour() else 'area {:.1f}'.format(self.ss_dtc.green_ccf.get_contour_area())
        tr = 'no' if not self.ss_dtc.red_ccf.has_contour() else 'area {:.1f}'.format(self.ss_dtc.red_ccf.get_contour_area())
        cv2.putText(out_img, 'start: {}'.format(tg), (20, 190), f, h, c, w)
        cv2.putText(out_img, 'finish: {}'.format(tr), (20, 240), f, h, c, w)
        return out_img
