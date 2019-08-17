#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server

import pdb

import two_d_guidance.trr.utils as trr_u
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se
import two_d_guidance.cfg.trr_state_estimatorConfig
import two_d_guidance.srv

class Node:

    def __init__(self, autostart=False):
        self.estimator = trr_se.StateEstimator(self.on_lm_passed)
        self.lane_model = trr_u.LaneModel()
        self.state_est_pub = trr_rpu.TrrStateEstimationPublisher('trr_state_est/status')
        # for now we just send one message - maybe we switch to a service call if that proves icky (lost message)
        self.start_crossed, self.finish_crossed = False, False
        srv_topic = 'LandmarkPassed'
        print('Waiting for service: {}'.format(srv_topic))
        rospy.wait_for_service(srv_topic)
        print '##service available'
        self.lm_crossed_srv_proxy = rospy.ServiceProxy(srv_topic, two_d_guidance.srv.LandmarkPassed)

        self.start_finish_sub = trr_rpu.TrrStartFinishSubscriber(user_callback=self.on_start_finish)
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()
        #odom_topic = '/caroline_robot_hardware/diff_drive_controller/odom' # real
        #odom_topic = '/caroline/diff_drive_controller/odom'                # sim
        odom_topic = '/odom'
        self.odom_sub = trr_rpu.OdomListener(odom_topic, 'state_estimator', 0.1, self.on_odom)
        self.lane_model_sub = trr_rpu.LaneModelSubscriber('/trr_vision/lane/detected_model', self.on_vision_lane)
        self.dyn_cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_state_estimatorConfig, self.dyn_cfg_callback)

    def dyn_cfg_callback(self, config, level):
        self.estimator.update_k_odom(config['k_odom'])
        return config
        
    def on_odom(self, msg):
        seq, stamp, vx, vy = msg.header.seq, msg.header.stamp, msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.estimator.update_odom(seq, stamp, vx, vy)

    def on_start_finish(self, msg):
        _, _, dist_to_start, dist_to_finish = self.start_finish_sub.get()
        self.estimator.update_landmarks(dist_to_start, dist_to_finish)

    def on_vision_lane(self, listener):
        print listener.get()
        
    def on_lm_passed(self, lm_id):
        rospy.loginfo('### on_lm_passed: passed {}'.format(self.estimator.path.lm_names[lm_id]))
        if   lm_id == self.estimator.path.LM_START:  self.start_crossed  = True
        elif lm_id == self.estimator.path.LM_FINISH: self.finish_crossed = True
        try:
            resp1 = self.lm_crossed_srv_proxy(lm_id)
        except rospy.ServiceException, e:
            print("Service call failed: {}".format(e))
            

        
    def periodic(self):
        self.state_est_pub.publish(self.estimator, self.start_crossed, self.finish_crossed)
        self.start_crossed, self.finish_crossed = False, False

        self.lane_model_sub.get(self.lane_model)

        #contour_start, contour_finish, dist_to_finish = self.start_finish_sub.get()
        #lred, lyellow, lgreen = self.traffic_light_sub.get()
        #print'red {} green {} finish {}'.format(lred, lgreen, dist_to_finish)
        #print('{}'.format(self.estimator.y))

    def run(self, periodic_freq=50.):
        rate = rospy.Rate(periodic_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass 

def main(args):
  rospy.init_node('trr_state_estimator_node')
  Node().run()

if __name__ == '__main__':
    main(sys.argv)
