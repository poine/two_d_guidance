#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic

import fl_utils as flu

class Node:

    def __init__(self):
        self.low_freq = 20
        self.robot_name = rospy.get_param('~robot_name', "pierrette")
        self.ref_frame = rospy.get_param('~ref_frame', '{}/base_link_footprint'.format(self.robot_name))
        self.fake_line_detector = flu.FakeLineDetector()
        self.lane_model = flu.LaneModel()
        self.lane_model_pub = flu.LaneModelPublisher()
        self.lane_model_marker_pub = flu.LaneModelMarkerPublisher(ref_frame=self.ref_frame,
                                                                  topic='/follow_line/detected_lane_fake', color=(1., 1., 0., 0.))
        
    def periodic(self):
        self.fake_line_detector.compute_line()
        if self.fake_line_detector.path_body is not None and len(self.fake_line_detector.path_body) > 3:
            self.lane_model.fit(self.fake_line_detector.path_body)
            self.lane_model_marker_pub.publish(self.lane_model)
            self.lane_model_pub.publish(self.lane_model)
    
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass



def main(args):
  rospy.init_node('fake_detector_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
