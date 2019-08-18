#!/usr/bin/env python
import os, sys
import math, numpy as np
import rospy, cv2

import pdb

import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se

# on the robot
# rosrun image_transport republish raw in:=/camera_road_front/image_raw compressed out:=/camera_road_front/image_raw/


class ImgPublisher(trr_rpu.DebugImgPublisher):
    def __init__(self, path, img_topic):
        trr_rpu.DebugImgPublisher.__init__(self, img_topic, '/trr_state_estimation/image_debug')
        self.s, self.c = 70., [500, 100] # scale and center of track display
        self.path_points = np.array([ self._world_to_img(p) for p in path.points]).astype(np.int)
        path_len = path.dists[-1] - path.dists[0]
        lm_start_idx, lm_start_point   = path.find_point_at_dist_from_idx(0, _d=path_len-1.21)
        lm_finish_idx, lm_finish_point = path.find_point_at_dist_from_idx(0, _d=0.86)
        self.lm_colors = [(0, 255, 0), (0, 0, 255)]
        self.lm_points = [tuple(self._world_to_img(lm_start_point).astype(np.int)), tuple(self._world_to_img(lm_finish_point).astype(np.int))]
        
    def _world_to_img(self, p_world):
        p_world =np.array([-p_world[0], p_world[1]])
        return self.s*p_world + self.c
        
    def _draw(self, img_bgr, model, path, y0=20, font_color=(0,255,255)):
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, font_color, 2
        cv2.putText(img_bgr, 'State Est', (y0, 40), f, h, c, w)
        cv2.polylines(img_bgr, [self.path_points], isClosed=True, color=(255, 0, 0), thickness=2)
        try:
            s_est, idx_s, v_est, dist_start, dist_to_finish = model.get()
        
            cv2.putText(img_bgr, 's: {:.2f}m ({})'.format(s_est, idx_s), (y0, 90), f, h, c, w)
            cv2.putText(img_bgr, 'v: {:.1f}m/s'.format(v_est), (y0, 140), f, h, c, w)
            #cv2.putText(img_bgr, 'lap: {:d}'.format(cur_lap), (y0, 190), f, h, c, w)
        
            p_est_idx, p_est = path.find_point_at_dist_from_idx(0, _d=s_est)
            if p_est is not None: # FIXME, used looped version
                cv2.circle(img_bgr, tuple(self._world_to_img(p_est).astype(np.int)), 5, (255,255,0), -1)
            for lm, _c in zip(self.lm_points, self.lm_colors):
                cv2.circle(img_bgr, lm, 5, _c, -1)
        except trr_rpu.NoRXMsgException :
            print('state estimator display: s.e. status NoRXMsgException')
        except trr_rpu.RXMsgTimeoutException :
            print('state estimator display: s.e. status RXMsgTimeoutException')
        
        

class Node:

    def __init__(self): #/caroline/camera_road_front
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        cam_name = rospy.get_param('~camera', prefix(robot_name, 'camera_road_front'))
        self.path = trr_se.StateEstPath('/home/poine/work/overlay_ws/src/two_d_guidance/paths/demo_z/track_trr_real.npz')
        self.im_pub = ImgPublisher(self.path, cam_name)
        self.state_est_sub = trr_rpu.TrrStateEstimationSubscriber(what='state est display')
        
    def periodic(self):
        self.im_pub.publish(self.state_est_sub, self.path)
     
    def run(self, freq=10):
        rate = rospy.Rate(freq)
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
