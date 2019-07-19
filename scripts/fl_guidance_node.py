#!/usr/bin/env python
import os, sys
import math, numpy as np
import roslib, rospy, rospkg, rostopic, dynamic_reconfigure.server
import nav_msgs.msg , geometry_msgs.msg, visualization_msgs.msg

import fl_utils as flu#, follow_line_node as fln
import two_d_guidance.srv
import two_d_guidance.cfg.fl_guidanceConfig

class Publisher:
    def __init__(self):
        ref_frame = rospy.get_param('~ref_frame', 'nono_0/base_link_footprint')
        self.pub_carrot = rospy.Publisher('pure_pursuit/goal', visualization_msgs.msg.Marker, queue_size=1)
        self.carrot_msg = visualization_msgs.msg.Marker()
        self.carrot_msg.header.frame_id=ref_frame
        self.carrot_msg.type = visualization_msgs.msg.Marker.CYLINDER
        p = self.carrot_msg.pose.position; p.x, p.y, p.z = 0, 0, 0.025
        o = self.carrot_msg.pose.orientation; o.x, o.y, o.z, o.w = 0, 0, 0, 1
        s = self.carrot_msg.scale; s.x, s.y, s.z = 0.01, 0.01, 0.1
        c = self.carrot_msg.color; c.a, c.r, c.g, c.b = 1., 0., 1., 0.

        self.pub_arc = rospy.Publisher('pure_pursuit/arc', nav_msgs.msg.Path, queue_size=1)
        self.arc_msg = nav_msgs.msg.Path()
        self.arc_msg.header.frame_id = ref_frame

        self.pub_lane = flu.LaneModelMarkerPublisher(topic='follow_line/detected_lane_model_guidance', color=(1, 1, 0, 0))
        
        cmd_topic = rospy.get_param('~cmd_topic', '/nono_0/diff_drive_controller/cmd_vel')
        rospy.loginfo(' publishing commands on: {}'.format(cmd_topic))
        self.pub_cmd = rospy.Publisher(cmd_topic, geometry_msgs.msg.Twist, queue_size=1)
 
    def publish_carrot(self, carrot_pos):
        self.carrot_msg.header.stamp = rospy.Time.now()
        p = self.carrot_msg.pose.position; p.x, p.y, p.z = carrot_pos[0], carrot_pos[1], 0
        self.pub_carrot.publish(self.carrot_msg)

    def publish_arc(self, R, carrot_pos):
        self.arc_msg.header.stamp = rospy.Time.now()
        self.arc_msg.poses = []
        alpha = np.arctan(carrot_pos[0]/(R - carrot_pos[1]))
        for theta in np.linspace(0, alpha, 20):
            pose = geometry_msgs.msg.PoseStamped()
            pose.pose.position.x =  R*np.sin(theta)   if not math.isinf(R) else 0
            pose.pose.position.y = -R*np.cos(theta)+R if not math.isinf(R) else 0
            self.arc_msg.poses.append(pose)
        self.pub_arc.publish(self.arc_msg)

    def publish_lane(self, lm):
        self.pub_lane.publish(lm)
        

    def publish_cmd(self, lin, ang):
        msg = geometry_msgs.msg.Twist()
        msg.linear.x, msg.angular.z = lin, ang
        self.pub_cmd.publish(msg)


        
        
class Guidance:
    mode_idle, mode_driving, mode_nb = range(3)
    def __init__(self, lookahead=0.4, vel_sp=0.2):
        self.set_mode(Guidance.mode_idle)
        self.lookahead = lookahead
        self.vel_sp = vel_sp
        self.carrot = [lookahead, 0]
        self.R = np.inf
    
    def compute(self, lane_model, lin=0.25, expl_noise=0.025):
        self.carrot = [self.lookahead, lane_model.get_y(self.lookahead)]
        self.R = (np.linalg.norm(self.carrot)**2)/(2*self.carrot[1])
        lin, ang = lin, lin/self.R
        ang += expl_noise*np.sin(0.5*rospy.Time.now().to_sec())
        return lin, ang

    def set_mode(self, mode):
        rospy.loginfo('guidance setting mode to {}'.format(mode))
        self.mode = mode
    
import pdb


class Node:

    def __init__(self):
        rospy.loginfo("fl_guidance_node Starting")
        self.low_freq = 10
        self.lane_model_sub = flu.LaneModelSubscriber()
        self.lane_model = flu.LaneModel()
        self.guidance = Guidance(lookahead=0.6)
        self.publisher = Publisher()
        #self.set_mode_svc = rospy.Service('set_mode', two_d_guidance.srv.SetMode, self.handle_set_mode)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.fl_guidanceConfig, self.cfg_callback)

    def cfg_callback(self, config, level):
        rospy.loginfo(" Reconfigure Request: mode: {guidance_mode}, lookahead: {lookahead}, vel_setpoint: {vel_sp}".format(**config))
        self.guidance.set_mode(config['guidance_mode'])
        self.guidance.lookahead = config['lookahead']
        self.guidance.vel_sp = config['vel_sp']
        return config

        
    # def handle_set_mode(self, req):
    #     print("Setting mode to {}".format(req.a))
    #     self.guidance.mode = req.a
    #     return two_d_guidance.srv.SetModeResponse(42)

    def periodic(self):
        self.lane_model_sub.get(self.lane_model)
        if self.guidance.mode == Guidance.mode_driving and self.lane_model.is_valid():
            lin, ang =  self.guidance.compute(self.lane_model, lin=self.guidance.vel_sp, expl_noise=0.)
        else:
            lin, ang = 0, 0
        self.publisher.publish_cmd(lin, ang)
        self.publisher.publish_arc(self.guidance.R, self.guidance.carrot)
        self.publisher.publish_carrot(self.guidance.carrot)
        self.publisher.publish_lane(self.lane_model)
    
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass



def main(args):
  rospy.init_node('follow_line_guidance_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
