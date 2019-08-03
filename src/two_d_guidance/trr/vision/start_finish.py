import numpy as np
import cv2
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_utils as trru


class StartFinishDetectPipeline(trr_vu.Pipeline):
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.ss_dtc = trr_vu.StartFinishDetector()
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.img = None
        self.green_ctr_lfp = None
        self.red_ctr_lfp = None
 
    def _process_image(self, img, cam):
        self.img = img
        self.ss_dtc.process_image(img)
        if self.ss_dtc.contour_green.cnt_max is not None:
            green_ctr_imp = cam.undistort_points(self.ss_dtc.contour_green.cnt_max.astype(np.float32))
            self.green_ctr_be = self.bird_eye.points_imp_to_be(green_ctr_imp)
            self.green_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.green_ctr_be)
            #M = cv2.moments(self.green_ctr_be)
            #print M
        else:
            self.green_ctr_lfp = None
        if self.ss_dtc.contour_red.cnt_max is not None:
            red_ctr_imp = cam.undistort_points(self.ss_dtc.contour_red.cnt_max.astype(np.float32))
            self.red_ctr_be = self.bird_eye.points_imp_to_be(red_ctr_imp)
            self.red_ctr_lfp = self.bird_eye.unwarped_to_fp(cam, self.red_ctr_be)
            # centroid
            M = cv2.moments(self.red_ctr_be); cx = M['m10']/M['m00']; cy = M['m01']/M['m00']
            #print cx, cy
            # centroid in local floor plane
            c_lfp = self.bird_eye.unwarped_to_fp(cam, np.array([[cx, cy], [cx, cy]]))[0]
            #print c_lfp
            self.dist_to_finish = np.linalg.norm(c_lfp)
            #print 'dist_to_finish {:.2f}m'.format(self.dist_to_finish)
            
        else:
            self.red_ctr_lfp = None
            

    def draw_debug(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
        out_img = self.ss_dtc.draw(self.img)
        self.draw_timing(out_img)
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2
        cds = [self.ss_dtc.contour_green, self.ss_dtc.contour_red]
        tg, tr = ['no' if _cd.cnt_max is None else '{}'.format(_cd.cnt_max_area) for _cd in cds] 
        cv2.putText(out_img, 'contours: {} {}'.format(tg, tr), (20, 190), f, h, c, w)
        return out_img
