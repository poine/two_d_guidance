#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic

import fl_utils as flu

class Node:

    def __init__(self):

        pass

    def periodic(self):
        pass

    
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass



def main(args):
  rospy.init_node('follow_line_node')
  rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
