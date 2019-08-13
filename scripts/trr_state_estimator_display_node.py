#!/usr/bin/env python
import os, sys
import math, numpy as np
import rospy, cv2

import pdb

import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se

class ImgPublisher(trr_rpu.DebugImgPublisher):
    def __init__(self, path):
        trr_rpu.DebugImgPublisher.__init__(self, '/caroline/camera_road_front', '/trr_state_estimation/image_debug')
        self.s, self.c = 70., [500, 100]
        self.path_points = np.array([ self._world_to_img(p) for p in path.points]).astype(np.int)
        
    def _world_to_img(self, p_world):
        p_world =np.array([-p_world[0], p_world[1]])
        return self.s*p_world + self.c
        
    def _draw(self, img_bgr, model, path, y0=20, font_color=(0,255,255)):
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, font_color, 2
        cv2.putText(img_bgr, 'State Est', (y0, 40), f, h, c, w)
        cv2.polylines(img_bgr, [self.path_points], isClosed=True, color=(255, 0, 0), thickness=2)
        s_est = model.get()
        if not math.isinf(s_est):
            p_est_idx, p_est = path.find_point_at_dist_from_idx(0, _d=s_est)
            #print s_est, p_est_idx, p_est
            cv2.circle(img_bgr, tuple(self._world_to_img(p_est).astype(np.int)), 5, (0,255,0), -1)
        cv2.putText(img_bgr, ' s: {:.2f}m'.format(s_est), (y0, 140), f, h, c, w)
        #pdb.set_trace()
        
        

class Node:

    def __init__(self):
        self.freq = 10
        self.path = trr_se.StateEstPath('/home/poine/work/overlay_ws/src/two_d_guidance/paths/demo_z/track_trr_real.npz')
        self.im_pub = ImgPublisher(self.path)
        self.ss_sub = trr_rpu.TrrStateEstimationSubscriber(what='state est display')
        
    def periodic(self):
        self.im_pub.publish(self.ss_sub, self.path)
     
    def run(self):
        rate = rospy.Rate(self.freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass


def main(args):
  rospy.init_node('trr_state_estimation_display_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
