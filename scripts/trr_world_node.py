#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server
import gazebo_msgs.msg

import pdb

import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.cfg.trr_worldConfig
import common_simulations.gazebo_traffic_light as gz_tlight
#
# Control the world (traffic place, etc)
#


class Node:

    def __init__(self, autostart=False, low_freq=1.):
        self.low_freq = low_freq
        self.tl = gz_tlight.TrafficLight(0.75, 2.25)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_worldConfig, self.cfg_callback)
        
    def cfg_callback(self, config, level):
        rospy.loginfo("  World Reconfigure Request:")
        m = rospy.wait_for_message('/gazebo/model_states', gazebo_msgs.msg.ModelStates)
        print m.name
        if config['traffic_light'] == self.tl.red:
            if 'traffic_light_green' in m.name : self.tl.switch_off(self.tl.green)
            self.tl.switch_on(self.tl.red)
            print('red')
        elif config['traffic_light'] == self.tl.green:
            if 'traffic_light_red' in m.name : self.tl.switch_off(self.tl.red)
            self.tl.switch_on(self.tl.green)
            print('green')

        return config
    
    def periodic(self):
        pass
    
    def run(self):
        if False:
            rate = rospy.Rate(self.low_freq)
            try:
                while not rospy.is_shutdown():
                    self.periodic()
                    rate.sleep()
            except rospy.exceptions.ROSInterruptException:
                pass
        else:
            rospy.spin()

        
def main(args):
  rospy.init_node('trr_world_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
