import numpy as np
import cv2
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.utils as trr_u



class Foo3Pipeline(trr_vu.Pipeline):
    show_none, show_input, show_mask, show_contour, show_be = range(5)
    def __init__(self, cam, be_param=trr_vu.BirdEyeParam()):
        trr_vu.Pipeline.__init__(self)
        self.bird_eye = trr_vu.BirdEyeTransformer(cam, be_param)
        self.line_finder = trr_vu.HoughLinesFinder()
        self.lane_model = trr_u.LaneModel()
        self.img = None
        self.undistorted_img = None
        self.display_mode = Foo3Pipeline.show_be
        
    def _process_image(self, img, cam):
        self.img = img
        self.undistorted_img = cam.undistort_img2(img)
        self.bird_eye.process(self.undistorted_img)
        self.line_finder.process(self.bird_eye.unwarped_img)
        if self.line_finder.has_lines():
            #print self.line_finder.lines.shape
            _all_pts = []
            for line in self.line_finder.vlines:
                x1,y1,x2,y2 = line.squeeze()
                _all_pts.append([x1, y1]); _all_pts.append([x2, y2])
            _all_pts_be = np.array(_all_pts)
            _all_pts_lfp = self.bird_eye.unwarped_to_fp(cam, _all_pts_be)
            #print _all_pts.shape
            self.lane_model.fit(_all_pts_lfp)
            self.lane_model.set_valid(True)
        else:
            self.lane_model.set_valid(False)

        
    def draw_debug(self, cam, img_cam=None):
        if self.img is None: return np.zeros((cam.h, cam.w, 3))
        if self.display_mode == Foo3Pipeline.show_input:
            debug_img = self.img
            #cv2.rectangle(debug_img, tuple(self.tl), tuple(self.br), color=(0, 0, 255), thickness=3)
        elif self.display_mode == Foo3Pipeline.show_mask:
            try:
                debug_img = cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR)
            except AttributeError:
                debug_img = np.zeros((cam.h, cam.w, 3))
        elif self.display_mode == Foo3Pipeline.show_contour:
            try:
                debug_img = cv2.cvtColor(self.line_finder.edges, cv2.COLOR_GRAY2BGR)
                self.line_finder.draw(debug_img)
            except AttributeError:
                debug_img = np.zeros((cam.h, cam.w, 3))
            debug_img = trr_vu.change_canvas(debug_img, cam.h, cam.w)
        elif self.display_mode == Foo3Pipeline.show_be:
            try:
                debug_img = self.bird_eye.unwarped_img
                if self.line_finder.has_lines():
                    self.line_finder.draw(debug_img)
                    self.bird_eye.draw_line(cam, debug_img, self.lane_model, self.lane_model.x_min, self.lane_model.x_max)
                debug_img = trr_vu.change_canvas(debug_img, cam.h, cam.w)
            except AttributeError:
                debug_img = np.zeros((cam.h, cam.w, 3))

        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 2
        cv2.putText(debug_img, 'Lane detection 3', (20, 40), f, h, c, w)
        self.draw_timing(debug_img, x0=350, y0=90)

        return cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)