#!/usr/bin/env python

'''
ackermann_drive_joyop.py:
    A ros joystick teleoperation script for ackermann steering based robots
'''

__author__ = 'George Kouros'
__license__ = 'GPLv3'
__maintainer__ = 'George Kouros'
__email__ = 'gkourosg@yahoo.gr'

import rospy
import ackermann_msgs.msg
from sensor_msgs.msg import Joy
import sys

class AckermannDriveJoyop:

    def __init__(self):
        cmd_topic = rospy.get_param('~cmd_topic', '/oscar_ackermann_controller/cmd_ack')
        self.max_speed = 1.
        self.max_steering_angle = 0.5
        self.axis_speed = rospy.get_param('~axis_linear', 1)
        self.axis_steering = rospy.get_param('~axis_steering', 2)
        self.enable_button = rospy.get_param('~enable_button', 4)

        rospy.loginfo('  Sending Ackermann messages to topic {}'.format(cmd_topic))
        self.speed = 0
        self.steering_angle = 0
        self.enabled = False
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
        self.drive_pub = rospy.Publisher(cmd_topic, ackermann_msgs.msg.AckermannDriveStamped,
                                         queue_size=1)
        rospy.Timer(rospy.Duration(1.0/50.0), self.pub_callback, oneshot=False)
        rospy.loginfo('ackermann_drive_joyop_node initialized')

    def joy_callback(self, joy_msg):
        #print joy_msg
        self.speed = joy_msg.axes[self.axis_speed] * self.max_speed;
        self.steering_angle = joy_msg.axes[self.axis_steering] * self.max_steering_angle;
        self.enabled = joy_msg.buttons[self.enable_button]

    def pub_callback(self, event):
        if self.enabled:
            ackermann_cmd_msg =  ackermann_msgs.msg.AckermannDriveStamped()
            ackermann_cmd_msg.header.stamp = rospy.Time.now()
            ackermann_cmd_msg.header.frame_id = 'odom'
            ackermann_cmd_msg.drive.speed = self.speed
            ackermann_cmd_msg.drive.steering_angle = self.steering_angle
            self.drive_pub.publish(ackermann_cmd_msg)
            #self.print_state()

    def print_state(self):
        sys.stderr.write('\x1b[2J\x1b[H')
        rospy.loginfo('\x1b[1M\r'
                      '\033[34;1mSpeed: \033[32;1m%0.2f m/s, '
                      '\033[34;1mSteering Angle: \033[32;1m%0.2f rad\033[0m',
                      self.speed, self.steering_angle)

    def finalize(self):
        rospy.loginfo('Halting motors, aligning wheels and exiting...')
        ackermann_cmd_msg = AckermannDrive()
        ackermann_cmd_msg.speed = 0
        ackermann_cmd_msg.steering_angle = 0
        self.drive_pub.publish(ackermann_cmd_msg)
        sys.exit()

if __name__ == '__main__':
    rospy.init_node('teleop_ackermann_joy')
    joyop = AckermannDriveJoyop()
    rospy.spin()
