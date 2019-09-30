#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server

import pdb

import two_d_guidance.trr.utils as trr_u
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se
import two_d_guidance.cfg.trr_state_estimatorConfig
import two_d_guidance.srv

class Node(trr_rpu.PeriodicNode):

    def __init__(self):
        name = 'trr_state_estimator_node'
        trr_rpu.PeriodicNode.__init__(self, name)

        tdg_dir = rospkg.RosPack().get_path('two_d_guidance')
        default_path_filename = os.path.join(tdg_dir, 'paths/demo_z/track_trr_real.npz')
        path_filename = rospy.get_param('~path_filename', default_path_filename)
        self.estimator = trr_se.StateEstimator(path_filename, self.on_lm_passed)
        self.lane_model = trr_u.LaneModel()
        self.state_est_pub = trr_rpu.TrrStateEstimationPublisher('trr_state_est/status')
        # we call race manager's  LandmarkPassed service to notify start/finish line crossing
        srv_topic = 'LandmarkPassed'
        print('Waiting for service: {}'.format(srv_topic))
        rospy.wait_for_service(srv_topic)
        print '##service available'
        self.lm_crossed_srv_proxy = rospy.ServiceProxy(srv_topic, two_d_guidance.srv.LandmarkPassed)

        
        self.start_finish_sub = trr_rpu.TrrStartFinishSubscriber(what=name, user_callback=self.on_start_finish)
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()
        #odom_topic = '/caroline_robot_hardware/diff_drive_controller/odom' # real
        #odom_topic = '/caroline/diff_drive_controller/odom'                # sim
        odom_topic = '/odom'                                                # external remaping
        self.odom_sub = trr_rpu.OdomListener(odom_topic, name, 0.1, self.on_odom)
        # unused for now
        #self.lane_model_sub = trr_rpu.LaneModelSubscriber('/trr_vision/lane/detected_model', 'state_estimator', user_cbk=self.on_vision_lane)

        self.dyn_cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_state_estimatorConfig, self.dyn_cfg_callback)
        # we expose a service for loading a velocity profile
        self.lm_service = rospy.Service('StateEstimatorLoadPath', two_d_guidance.srv.GuidanceLoadVelProf, self.on_load_path)

    def on_load_path(self, req):
        path_filename = ''.join(req.path_filename)
        print('on_load_path {}'.format(path_filename))
        self.estimator.load_path(path_filename)
        err = 0
        return two_d_guidance.srv.GuidanceLoadVelProfResponse(err)
    
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
        #print listener.get()
        pass
        
    def on_lm_passed(self, lm_id):
        rospy.loginfo('### on_lm_passed: passed {}'.format(self.estimator.path.lm_names[lm_id]))
        try:
            resp1 = self.lm_crossed_srv_proxy(lm_id)
        except rospy.ServiceException, e:
            print("Service call failed: {}".format(e))


    def periodic(self):
        self.state_est_pub.publish(self.estimator)
        # not yet used
        # try:
        #     self.lane_model_sub.get(self.lane_model)
        # except trr_rpu.NoRXMsgException:
        #     self.lane_model.set_valid(False)
        #     rospy.loginfo_throttle(1., 'trr_state_estimation lane_model: NoRXMsgException')
        # except trr_rpu.RXMsgTimeoutException:
        #     self.lane_model.set_valid(False)
        #     rospy.loginfo_throttle(1., 'trr_state_estimation lane_model: RXMsgTimeoutException')



def main(args):
    Node().run(freq=50)

if __name__ == '__main__':
    main(sys.argv)
