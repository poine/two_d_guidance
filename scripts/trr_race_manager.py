#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
#from geometry_msgs.msg import PolygonStamped, Point32
import pdb

import two_d_guidance.trr_rospy_utils as trr_rpu

mode_staging, mode_racing, mode_finished = range(3)

class Node:

    def __init__(self, autostart=True):
        self.low_freq = 20
        #self.pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)
        self.start_finish_sub = trr_rpu.TrrStartFinishSubscriber()
        
        client_name = "trr_guidance"
        self.cfg_client = dynamic_reconfigure.client.Client(client_name, timeout=30, config_callback=self.guidance_cfg_callback)
        rospy.loginfo(' client_name: {}'.format(client_name))
        if autostart:
            self.mode = mode_racing
            self.set_guidance_mode(2)
        else:
            self.mode = mode_staging
            self.set_guidance_mode(0)
                             
    def guidance_cfg_callback(self, config):
        if config.guidance_mode == 2:
            self.mode = mode_racing
        #rospy.loginfo("Config set {}".format(config))
        pass
    
    def periodic(self):
        start_points, finish_points, dist_to_finish = self.start_finish_sub.get()
        if len(finish_points) > 0:
            rospy.loginfo('Viewing finish {}'.format(dist_to_finish))
            if self.mode == mode_racing and dist_to_finish < 0.15:
                self.set_guidance_mode(1)
                self.mode = mode_finished
                

    def set_guidance_mode(self, mode):
        rospy.loginfo(' set guidance mode to {}'.format(mode))
        self.cfg_client.update_configuration({"guidance_mode":mode})
            
        
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

        
def main(args):
  rospy.init_node('race_manager_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
