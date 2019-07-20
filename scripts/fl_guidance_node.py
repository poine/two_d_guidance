#!/usr/bin/env python
import os, sys
import math, numpy as np
import roslib, rospy, rospkg, rostopic, dynamic_reconfigure.server
import nav_msgs.msg , geometry_msgs.msg, visualization_msgs.msg, sensor_msgs.msg

import fl_utils as flu
#import two_d_guidance.srv
import two_d_guidance.cfg.fl_guidanceConfig  # dynamic reconfigure
import smocap.rospy_utils, cv2               # we publish a debug image

import pdb

class Publisher:
    def __init__(self, ref_frame):
        self.pub_carrot = rospy.Publisher('trr_guidance/goal', visualization_msgs.msg.Marker, queue_size=1)
        self.carrot_msg = visualization_msgs.msg.Marker()
        self.carrot_msg.header.frame_id=ref_frame
        self.carrot_msg.type = visualization_msgs.msg.Marker.CYLINDER
        p = self.carrot_msg.pose.position; p.x, p.y, p.z = 0, 0, 0.025
        o = self.carrot_msg.pose.orientation; o.x, o.y, o.z, o.w = 0, 0, 0, 1
        s = self.carrot_msg.scale; s.x, s.y, s.z = 0.01, 0.01, 0.1
        c = self.carrot_msg.color; c.a, c.r, c.g, c.b = 1., 0., 1., 0.

        self.pub_arc = rospy.Publisher('trr_guidance/arc', nav_msgs.msg.Path, queue_size=1)
        self.arc_msg = nav_msgs.msg.Path()
        self.arc_msg.header.frame_id = ref_frame

        self.pub_lane = flu.LaneModelMarkerPublisher(ref_frame=ref_frame, topic='trr_guidance/detected_lane_model_guidance', color=(1, 1, 0, 0))
        
        cmd_topic = rospy.get_param('~cmd_topic', '/nono_0/diff_drive_controller/cmd_vel')
        rospy.loginfo(' publishing commands on: {}'.format(cmd_topic))
        self.pub_cmd = rospy.Publisher(cmd_topic, geometry_msgs.msg.Twist, queue_size=1)

        self.img_pub = ImgPublisher()
        
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

class ImgPublisher:
    def __init__(self, img_topic = "/trr_guidance/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic+"/compressed", sensor_msgs.msg.CompressedImage, queue_size=1)
        cam_names = ['caroline/camera_road_front']
        self.img = None
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)
        
    def on_image(self, img, (cam_idx, stamp, seq)):
        # store subscribed image (bgr, opencv)
        self.img = img
    
    def publish(self, lin_sp, lin_odom, ang_sp, ang_odom):
        if self.img is not None:
            self.draw(lin_sp, lin_odom, ang_sp, ang_odom)
            img_rgb = self.img[...,::-1] # rgb = bgr[...,::-1] OpenCV image to Matplotlib
            msg = sensor_msgs.msg.CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img_rgb)[1]).tostring()
            self.image_pub.publish(msg)

    def draw(self, lin_sp, lin_odom, ang_sp, ang_odom):
        cv2.putText(self.img, 'lin:  sp/odom {:.2f}/{:.2f} m/s'.format(lin_sp, lin_odom), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2)
        cv2.putText(self.img, 'ang: sp/odom {: 6.2f}/{: 6.2f} deg/s'.format(np.rad2deg(ang_sp), np.rad2deg(ang_odom)), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2)


class OdomListener:
    def __init__(self, odom_topic = '/caroline/diff_drive_controller/odom'):
        self.odom_sub = rospy.Subscriber(odom_topic, nav_msgs.msg.Odometry, self.callback)
        self.lin, self.ang = 0, 0
        
    def callback(self, msg):
        self.msg = msg
        self.lin, self.ang = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        
        
class Guidance:
    mode_idle, mode_stopped, mode_driving, mode_nb = range(4)
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
        self.publisher = Publisher(ref_frame)
        #self.set_mode_svc = rospy.Service('set_mode', two_d_guidance.srv.SetMode, self.handle_set_mode)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.fl_guidanceConfig, self.cfg_callback)
        self.lane_model_sub = flu.LaneModelSubscriber()
        self.odom_sub = OdomListener()

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
        if self.guidance.mode != Guidance.mode_idle:
            if self.guidance.mode == Guidance.mode_driving and self.lane_model.is_valid():
                self.lin_sp, self.ang_sp =  self.guidance.compute(self.lane_model, lin=self.guidance.vel_sp, expl_noise=0.)
            else:
                self.lin_sp, self.ang_sp = 0, 0
            self.publisher.publish_cmd(self.lin_sp, self.ang_sp)
        self.hf_loop_idx += 1
        self.low_freq()
        
    def low_freq(self):
        i = self.hf_loop_idx%self.low_freq_div
        steps = [ lambda : self.publisher.publish_arc(self.guidance.R, self.guidance.carrot),
                  lambda : self.publisher.publish_carrot(self.guidance.carrot),
                  lambda : self.publisher.img_pub.publish(self.lin_sp, self.odom_sub.lin, self.ang_sp, self.odom_sub.ang),
                  lambda : self.publisher.publish_lane(self.lane_model),
                  lambda : self.publisher.img_pub.publish(self.lin_sp, self.odom_sub.lin, self.ang_sp, self.odom_sub.ang),
                  lambda : None ]
        steps[i]()
        
    def run(self):
        rate = rospy.Rate(self.high_freq)
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
