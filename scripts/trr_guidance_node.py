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
        self.sta_pub = trr_rpu.GuidanceStatusPublisher(topic, 'guidance')
        self.pub_cmd = rospy.Publisher(cmd_topic, geometry_msgs.msg.Twist, queue_size=1)

    def publish_cmd(self, lin, ang):
        msg = geometry_msgs.msg.Twist()
        msg.linear.x, msg.angular.z = lin, ang
        self.pub_cmd.publish(msg)
        
    def publish_status(self, guidance): self.sta_pub.publish(guidance)


class Node(trr_rpu.PeriodicNode):

    def __init__(self):
        trr_rpu.PeriodicNode.__init__(self, 'trr_guidance_node')
        rospy.loginfo("trr_guidance_node Starting")
        ref_frame = rospy.get_param('~ref_frame', 'caroline/base_link_footprint')
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))

        tdg_dir = rospkg.RosPack().get_path('two_d_guidance')
        path_name = rospy.get_param('~path_name', 'track_trr_real_vel_1.npz')
        fname = os.path.join(tdg_dir, 'paths/demo_z/{}'.format(path_name))
        lookahead = rospy.get_param('~lookahead', 0.8)
        self.guidance = trr_gui.Guidance(lookahead=lookahead, path_fname=fname)

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
        try:
            self.lane_model_sub.get(self.guidance.lane)
            _s, _is, _v, _ds, _df = self.state_est_sub.get()
            self.guidance.compute(_s, _is, expl_noise=0.)
            if self.guidance.mode != trr_gui.Guidance.mode_idle:
                self.publisher.publish_cmd(self.guidance.lin_sp, self.guidance.ang_sp)
            self.publisher.publish_status(self.guidance)
        except trr_rpu.NoRXMsgException:
            rospy.loginfo_throttle(1., 'guidance (lane or state_est): NoRXMsgException')
        except trr_rpu.RXMsgTimeoutException:
            rospy.loginfo_throttle(1., 'guidance: (lane or state_est) RXMsgTimeoutException')


def main(args):
  Node().run(30)

if __name__ == '__main__':
    main(sys.argv)
