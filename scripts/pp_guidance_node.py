#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import nav_msgs.msg , geometry_msgs.msg, ackermann_msgs.msg, visualization_msgs.msg
import math, numpy as np

import two_d_guidance as tdg
import two_d_guidance.ros_utils as ros_utils

#
# rewrite of pure pursuit
#
# TODO: fix debug publishing

class NodePublisher:
    def __init__(self, ref_link):
        self.pub_path = rospy.Publisher('pure_pursuit/path', nav_msgs.msg.Path, queue_size=1)
        self.pub_goal = rospy.Publisher('pure_pursuit/goal', visualization_msgs.msg.Marker, queue_size=1)
        self.pub_arc  = rospy.Publisher('pure_pursuit/arc', nav_msgs.msg.Path, queue_size=1)

        self.goal_msg = visualization_msgs.msg.Marker()
        self.goal_msg.header.frame_id="world"
        self.goal_msg.type = visualization_msgs.msg.Marker.CYLINDER
        o = self.goal_msg.pose.orientation; o.x, o.y, o.z, o.w = 0, 0, 0, 1
        s = self.goal_msg.scale; s.x, s.y, s.z = 0.01, 0.01, 0.1
        c = self.goal_msg.color; c.a, c.r, c.g, c.b = 1., 1., 1., 1.
        
        self.path_msg = nav_msgs.msg.Path()
        self.path_msg.header.frame_id = ref_link#"nono_0/base_link"#"root_link_m0_actual"

        
    def publish_path(self, _path):
        path_msg = nav_msgs.msg.Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id="world"
        for l in _path.points:
            pose = geometry_msgs.msg.PoseStamped()
            pose.pose.position.x, pose.pose.position.y = l
            path_msg.poses.append(pose)
        self.pub_path.publish(path_msg)

    def publish_debug(self, goal_pos, R):
        self.goal_msg.header.stamp = rospy.Time.now()
        p = self.goal_msg.pose.position; p.x, p.y, p.z = goal_pos[0], goal_pos[1], 0
        self.pub_goal.publish(self.goal_msg)
        
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses = []
        for theta in np.arange(0, 2*math.pi, 0.01):
            pose = geometry_msgs.msg.PoseStamped()
            pose.pose.position.x =  R*math.sin(theta)
            pose.pose.position.y = -R*math.cos(theta)+R
            self.path_msg.poses.append(pose)
        self.pub_arc.publish(self.path_msg)


class VelSetpointCst:
    def __init__(self, _v):
        self.v = _v

    def get(self, t):
          return self.v #+ self.v/2*np.sin(0.1*t)

        
class Node:
    def __init__(self):

        twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/oscar_ackermann_controller/cmd_vel')
        ack_cmd_topic = rospy.get_param('~ack_cmd_topic', None)
        if ack_cmd_topic is not None:
            self.publish_ack_cmd = True
            self.pub_ack = rospy.Publisher(ack_cmd_topic, ackermann_msgs.msg.AckermannDriveStamped, queue_size=1)
            rospy.loginfo(' publishing ack commands on: {}'.format(ack_cmd_topic))
        else:
            self.publish_ack_cmd = False
            rospy.loginfo(' publishing twist commands on: {}'.format(twist_cmd_topic))
            self.pub_twist = rospy.Publisher(twist_cmd_topic, geometry_msgs.msg.Twist, queue_size=1)
            
        path_filename = rospy.get_param('~path_filename',
                          os.path.join(rospkg.RosPack().get_path('two_d_guidance'), 'paths/demo_z/track_ethz_cam1_new.npz'))
        param = tdg.pure_pursuit.Param()
        self.l = param.L = 0.08
        self.ctl = tdg.pure_pursuit.PurePursuit(path_filename, param)
        self.v_sp = rospy.get_param('~vel_setpoint', 0.5)
        self.v_ctl = VelSetpointCst(self.v_sp)

        
        self.robot_pose_topic = rospy.get_param('~robot_pose_topic', None)
        msg_class, _msg_topic, _unused = rostopic.get_topic_class(self.robot_pose_topic, blocking=True)
        if msg_class == geometry_msgs.msg.PoseWithCovarianceStamped:
            print 'using geometry_msgs.msg.PoseWithCovarianceStamped for robot location'
            self.robot_listener = ros_utils.SmocapListener(topic=self.robot_pose_topic)
        elif msg_class == nav_msgs.msg.Odometry:
            print 'using nav_msgs.msg.Odometry for robot location'
            self.robot_listener = ros_utils.GazeboTruthListener(topic=self.robot_pose_topic)
        else:
            print 'unsupported robot location message type'
            rospy.signal_shutdown('unsupported robot location message class')


        self.robot_ref_link = rospy.get_param('~robot_ref_link', 'base_link')
        rospy.loginfo(' getting robot pose from: {} ({})'.format(self.robot_pose_topic, msg_class))   
        rospy.loginfo(' robot reference link: {} '.format(self.robot_ref_link))
        
        rospy.loginfo(' loading path: {}'.format(path_filename))
        rospy.loginfo('   velocity setpoint: {} m/s'.format(self.v_sp))
        rospy.loginfo('   wheels_kinematic_l: {} m'.format(self.l))

        self.node_pub = NodePublisher(self.robot_ref_link)

        self.node_sub = rospy.Subscriber('pure_pursuit/vel_setpoint', geometry_msgs.msg.Twist, self.vel_sp_cbk)
        
    def vel_sp_cbk(self, msg):
        rospy.loginfo_throttle(0.5, 'vel sp {}'.format(msg.linear.x))
        self.v_ctl.v = msg.linear.x
        
    def periodic(self):
        try:
            # get current pose
            p0, psi = self.robot_listener.get_loc_and_yaw()
            _unused, self.alpha = self.ctl.compute_looped(p0, psi)
            self.alpha += 0.025*np.sin(0.5*rospy.Time.now().to_sec())
            self.v = self.v_ctl.get(rospy.Time.now().to_sec())
            self.publish_command()
        except ros_utils.RobotLostException:
             rospy.loginfo_throttle(0.5, 'robot lost')
        except ros_utils.RobotNotLocalizedException:
            rospy.loginfo_throttle(1., "Robot not localized") # print every second
        self.node_pub.publish_path(self.ctl.path) # expensive... 50% CPU
        self.node_pub.publish_debug(self.ctl.p2, self.ctl.R) 

    def publish_command(self):
            if self.publish_ack_cmd:
                self.publish_ackermann_cmd()
            else:
                self.publish_twist_cmd()
        
    def publish_twist_cmd(self):
        lin = self.v
        ang = self.v/self.l*math.tan(self.alpha)
        msg = geometry_msgs.msg.Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.pub_twist.publish(msg)

    def publish_ackermann_cmd(self):
        msg = ackermann_msgs.msg.AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'odom'
        msg.drive.steering_angle = self.alpha
        msg.drive.speed = self.v
        self.pub_ack.publish(msg)
        
    def run(self):
        self.rate = rospy.Rate(20.)
        while not rospy.is_shutdown():
            self.periodic()
            self.rate.sleep()
        

def main(args):
  rospy.init_node('pp_guidance')
  Node().run()

  
if __name__ == '__main__':
    main(sys.argv)

