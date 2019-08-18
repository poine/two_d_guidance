#!/usr/bin/env python
import os, sys
import math, numpy as np
import roslib, rospy, rospkg, rostopic, dynamic_reconfigure.server
import nav_msgs.msg , geometry_msgs.msg#, visualization_msgs.msg, sensor_msgs.msg

import two_d_guidance.trr.utils as trr_u
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.guidance as trr_gui
import two_d_guidance.cfg.trr_guidanceConfig  # dynamic reconfigure

import pdb


        



class Publisher:
    def __init__(self, topic='trr_guidance/status', cmd_topic='trr_guidance/cmd'):
        self.sta_pub = trr_rpu.SimplePublisher(topic, two_d_guidance.msg.FLGuidanceStatus, "guidance status")
        self.pub = rospy.Publisher(topic, two_d_guidance.msg.FLGuidanceStatus, queue_size=1)
        self.pub_cmd = rospy.Publisher(cmd_topic, geometry_msgs.msg.Twist, queue_size=1)

    def publish_cmd(self, lin, ang):
        msg = geometry_msgs.msg.Twist()
        msg.linear.x, msg.angular.z = lin, ang
        self.pub_cmd.publish(msg)
        
    def publish_status(self, guidance, lane_model, lin_sp, ang_sp):
        msg = two_d_guidance.msg.FLGuidanceStatus()
        msg.guidance_mode = guidance.mode
        msg.poly = lane_model.coefs
        msg.lookahead_dist = guidance.lookahead_dist
        msg.lookahead_time = guidance.lookahead_time
        msg.carrot_x, msg.carrot_y = guidance.carrot
        msg.R = guidance.R
        msg.lin_sp, msg.ang_sp = lin_sp, ang_sp
        self.pub.publish(msg)


class Node:

    def __init__(self):
        rospy.loginfo("fl_guidance_node Starting")
        ref_frame = rospy.get_param('~ref_frame', 'caroline/base_link_footprint')
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))
        self.hf_loop_idx, self.low_freq_div = 0, 6
        self.lin_sp, self.ang_sp = 0.,0.
        self.lane_model = trr_u.LaneModel()

        tdg_dir = rospkg.RosPack().get_path('two_d_guidance')
        fname = os.path.join(tdg_dir, 'paths/demo_z/track_trr_real_vel.npz')
        self.guidance = trr_gui.Guidance(lookahead=0.6, path_fname=fname)

        cmd_topic = rospy.get_param('~cmd_topic', '/caroline/diff_drive_controller/cmd_vel')
        rospy.loginfo(' publishing commands on: {}'.format(cmd_topic))
        self.publisher = Publisher(cmd_topic=cmd_topic)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_guidanceConfig, self.cfg_callback)
        self.lane_model_sub = trr_rpu.LaneModelSubscriber('/trr_vision/lane/detected_model')
        self.state_est_sub = trr_rpu.TrrStateEstimationSubscriber(what='guidance')
        

    def cfg_callback(self, config, level):
        rospy.loginfo(" Reconfigure Request: mode: {guidance_mode}, lookahead: {lookahead_dist}/{lookahead_time}, vel_setpoint: {vel_sp}".format(**config))
        self.guidance.set_mode(config['guidance_mode'])
        self.guidance.lookaheads[0].set_dist(config['lookahead_dist'])
        self.guidance.lookaheads[1].set_time(config['lookahead_time'])
        self.guidance.lookahead_mode = config['lookahead_mode']
        self.guidance.vel_ctl.sp = config['vel_sp']
        self.guidance.vel_ctl.k_curv = config['vel_k_curve']
        self.guidance.vel_ctl.mode = config['vel_ctl_mode']
        return config


    def periodic(self):
        self.lane_model_sub.get(self.lane_model)
        try:
            _s, _is, _v, _ds, _df = self.state_est_sub.get()
            if self.guidance.mode != trr_gui.Guidance.mode_idle:
                if self.guidance.mode == trr_gui.Guidance.mode_driving and self.lane_model.is_valid():
                    self.lin_sp, self.ang_sp =  self.guidance.compute(self.lane_model, _s, _is, expl_noise=0.)
                else:
                    self.lin_sp, self.ang_sp = 0, 0
                self.publisher.publish_cmd(self.lin_sp, self.ang_sp)
            self.publisher.publish_status(self.guidance, self.lane_model, self.lin_sp, self.ang_sp)
        except trr_rpu.NoRXMsgException:
            rospy.loginfo_throttle(1., 'guidance: NoRXMsgException')
        except trr_rpu.RXMsgTimeoutException:
            rospy.loginfo_throttle(1., 'guidance: RXMsgTimeoutException')
        self.hf_loop_idx += 1

        #self.low_freq()
        
    # def low_freq(self):
    #     pass
    
    def run(self, high_freq=30):
        rate = rospy.Rate(high_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

def main(args):
  rospy.init_node('trr_guidance_node')
  Node().run()

if __name__ == '__main__':
    main(sys.argv)
