#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
#from geometry_msgs.msg import PolygonStamped, Point32
import pdb

import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se


class Node:

    def __init__(self, autostart=False, low_freq=10.):
        self.low_freq = low_freq
        self.start_finish_sub = trr_rpu.TrrStartFinishSubscriber()
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()
        self.estimator = trr_se.StateEstimator()

    def periodic(self):
        contour_start, contour_finish, dist_to_finish = self.start_finish_sub.get()
        lred, lyellow, lgreen = self.traffic_light_sub.get()
        print'red {} green {} finish {}'.format(lred, lgreen, dist_to_finish)
        print('{}'.format(self.estimator.y))

     

    def run(self):
        rate = rospy.Rate(self.low_freq)
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
