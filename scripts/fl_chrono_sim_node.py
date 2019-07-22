#!/usr/bin/env python
import os, sys
import math, numpy as np
import roslib, rospy

import two_d_guidance.ros_utils as ros_utils

class Chrono:
    mode_stopped, mode_lap = range(2)
    def __init__(self):
        self.start_x = -1.25
        self.end_x = 1.25
        self.mode = Chrono.mode_stopped
        self.track_len = 17.55 + 2.5 # len of track in meter
        
    def start(self):
        pass
    
        
    def run(self, past_loc, cur_loc):
        if past_loc is not None:
            if past_loc[0] < self.start_x  and cur_loc[0] >= self.start_x:
                #print 'starting: ', past_loc, cur_loc
                if self.mode == Chrono.mode_stopped:
                    print 'lap start'
                    self.start = rospy.Time.now()
                    self.mode = Chrono.mode_lap
                    self.end_cnt = 0
                    
            elif past_loc[0] < self.end_x  and cur_loc[0] >= self.end_x:
                #print 'ending: ', past_loc, cur_loc
                if self.mode == Chrono.mode_lap:
                    self.end_cnt += 1
                    if self.end_cnt >= 2:
                        self.end = rospy.Time.now()
                        self.mode = Chrono.mode_stopped
                        lap_duration = (self.end-self.start).to_sec()
                        lap_vel = self.track_len / lap_duration
                        print('lap {:.2f}s ({:.1f} m/s)'.format(lap_duration, lap_vel))
                    
    
class Node:
    def __init__(self):
        rospy.loginfo("trr_chrono_sim_node Starting")
        self.freq = 10.
        self.prev_pose = None
        self.chrono = Chrono()
        self.robot_pose_topic = rospy.get_param('~robot_pose_topic', "/caroline/base_link_truth1")
        self.robot_listener = ros_utils.GazeboTruthListener(topic=self.robot_pose_topic)
        rospy.loginfo(' getting robot pose from: {}'.format(self.robot_pose_topic))
        
    def periodic(self):
        try:
            p0, psi = self.robot_listener.get_loc_and_yaw()
            self.chrono.run(self.prev_pose, p0)
            self.prev_pose = p0
        except ros_utils.RobotNotLocalizedException:
            rospy.loginfo_throttle(1., "Robot not localized") # print every second
        except ros_utils.RobotLostException:
             rospy.loginfo_throttle(0.5, 'robot lost')

    def run(self):
        rate = rospy.Rate(self.freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
    
        
        
def main(args):
  rospy.init_node('trr_chrono_sim_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
