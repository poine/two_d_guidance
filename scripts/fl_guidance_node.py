#!/usr/bin/env python
import os, sys
import math, numpy as np
import roslib, rospy, rospkg, rostopic, dynamic_reconfigure.server
import nav_msgs.msg , geometry_msgs.msg#, visualization_msgs.msg, sensor_msgs.msg

import fl_utils as flu
import two_d_guidance.cfg.fl_guidanceConfig  # dynamic reconfigure

import pdb


class OdomListener:
    def __init__(self, odom_topic = '/caroline/diff_drive_controller/odom'):
        self.odom_sub = rospy.Subscriber(odom_topic, nav_msgs.msg.Odometry, self.callback)
        self.lin, self.ang = 0, 0
        
    def callback(self, msg):
        self.msg = msg
        self.lin, self.ang = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        

class VelCtl:
    mode_cst, mode_curv, mode_nb = range(3)
    def __init__(self):
        self.mode = VelCtl.mode_cst#VelCtl.mode_curv#
        self.sp = 1.
        self.min_sp = 0.2
        self.k_curv = 0.5

    def get(self, lane_model):
        if self.mode == VelCtl.mode_cst:
            return self.sp
        else:
            curv = lane_model.coefs[1]
            return max(self.min_sp, self.sp-np.abs(curv)*self.k_curv)

class CstLookahead:
    def __init__(self, _d=0.4): self.d = _d
    def set_dist(self, _d): self.d = _d
    def set_time(self, _t): pass
    def get_dist(self, _v): return self.d

class TimeCstLookahead:
    def __init__(self, _t=0.4): self.t = _t
    def set_dist(self, _d): pass
    def set_time(self, _t): self.t = _t
    def get_dist(self, _v): return _v*self.t
    
        
class Guidance:
    mode_idle, mode_stopped, mode_driving, mode_nb = range(4)
    mode_lookahead_dist, mode_lookahead_time = range(2)
    def __init__(self, lookahead=0.4, vel_sp=0.2):
        self.set_mode(Guidance.mode_idle)
        self.lookaheads = [CstLookahead(), TimeCstLookahead()]
        self.lookahead_mode = Guidance.mode_lookahead_dist
        self.lookahead_dist = 0.1
        self.lookahead_time = 0.1
        self.carrot = [self.lookahead_dist, 0]
        self.R = np.inf
        self.vel_ctl = VelCtl()
        self.vel_sp = vel_sp
    
    def compute(self, lane_model, expl_noise=0.025):
        lin = self.vel_ctl.get(lane_model)
        self.lookahead_dist = self.lookaheads[self.lookahead_mode].get_dist(lin)
        self.lookahead_time = np.inf if lin == 0 else self.lookahead_dist/lin
        self.carrot = [self.lookahead_dist, lane_model.get_y(self.lookahead_dist)]
        self.R = (np.linalg.norm(self.carrot)**2)/(2*self.carrot[1])
        lin, ang = lin, lin/self.R
        ang += expl_noise*np.sin(0.5*rospy.Time.now().to_sec())
        return lin, ang

    def set_mode(self, mode):
        rospy.loginfo('guidance setting mode to {}'.format(mode))
        self.mode = mode
    

class Publisher:
    def __init__(self, topic='trr_guidance/status', cmd_topic='trr_guidance/cmd'):
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
        ref_frame = rospy.get_param('~ref_frame', 'nono_0/base_link_footprint')
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))
        self.high_freq = 30
        self.hf_loop_idx = 0
        self.low_freq_div = 6
        self.lin_sp, self.ang_sp = 0,0
        self.lane_model = flu.LaneModel()
        self.guidance = Guidance(lookahead=0.6)
        cmd_topic = rospy.get_param('~cmd_topic', '/nono_0/diff_drive_controller/cmd_vel')
        rospy.loginfo(' publishing commands on: {}'.format(cmd_topic))
        self.publisher = Publisher(cmd_topic=cmd_topic)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.fl_guidanceConfig, self.cfg_callback)
        self.lane_model_sub = flu.LaneModelSubscriber()
        self.odom_sub = OdomListener()

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
        if self.guidance.mode != Guidance.mode_idle:
            if self.guidance.mode == Guidance.mode_driving and self.lane_model.is_valid():
                self.lin_sp, self.ang_sp =  self.guidance.compute(self.lane_model, expl_noise=0.)
            else:
                self.lin_sp, self.ang_sp = 0, 0
            self.publisher.publish_cmd(self.lin_sp, self.ang_sp)
        self.hf_loop_idx += 1
        self.publisher.publish_status(self.guidance, self.lane_model, self.lin_sp, self.ang_sp)
        #self.low_freq()
        
    # def low_freq(self):
    #     i = self.hf_loop_idx%self.low_freq_div
    #     steps = [ lambda : self.publisher.publish_arc(self.guidance.R, self.guidance.carrot),
    #               lambda : self.publisher.publish_carrot(self.guidance.carrot),
    #               lambda : self.publisher.img_pub.publish(self.guidance.mode, self.lin_sp, self.odom_sub.lin, self.ang_sp, self.odom_sub.ang, self.lane_model, self.guidance.lookahead_dist),
    #               lambda : self.publisher.publish_lane(self.lane_model),
    #               lambda : self.publisher.img_pub.publish(self.guidance.mode, self.lin_sp, self.odom_sub.lin, self.ang_sp, self.odom_sub.ang, self.lane_model, self.guidance.lookahead_dist),
    #               lambda : None ]
    #     steps[i]()
        
    def run(self):
        rate = rospy.Rate(self.high_freq)
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
