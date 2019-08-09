#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
#from geometry_msgs.msg import PolygonStamped, Point32
import pdb

import two_d_guidance.trr_rospy_utils as trr_rpu
import two_d_guidance.cfg.trr_race_managerConfig


class Node:
    mode_staging, mode_ready, mode_racing, mode_finished = range(4)
    def __init__(self, autostart=False, low_freq=20.):
        self.low_freq = low_freq
        self.start_finish_sub = trr_rpu.TrrStartFinishSubscriber()
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()

        self.guidance_cfg_client, self.race_manager_cfg_srv = None, None 
        # we will expose some parameters to users
        self.race_manager_cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_race_managerConfig, self.race_manager_cfg_callback)
        # and manipulate parameters exposed by the guidance node
        guidance_client_name = "trr_guidance_node"
        self.guidance_cfg_client = dynamic_reconfigure.client.Client(guidance_client_name, timeout=30, config_callback=self.guidance_cfg_callback)
        rospy.loginfo(' guidance_client_name: {}'.format(guidance_client_name))

        self.update_race_mode(Node.mode_racing if autostart else Node.mode_staging)

    def update_race_mode(self, mode):
        if self.race_manager_cfg_srv is not None:
            self.race_manager_cfg_srv.update_configuration({'mode': mode})

    def enter_staging(self):
        self.set_guidance_mode(0) # idle when staging
        self.cur_lap = 0

    def enter_racing(self):
        self.set_guidance_mode(2) # driving when racing
        
            
    # reconfigure (dyn config) race_mode    
    def set_race_mode(self, mode):
        self.mode = mode
        if mode == Node.mode_racing:
            #self.set_guidance_mode(2) # driving when racing
            self.enter_racing()
        elif mode == Node.mode_staging:
            #self.set_guidance_mode(0) # idle when staging
            self.enter_staging()
        elif mode == Node.mode_ready or mode == Node.mode_finished:
            self.set_guidance_mode(1) # stoped otherwise
            

        
    def race_manager_cfg_callback(self, config, level):
        rospy.loginfo("  Race Manager Reconfigure Request:")
        #pdb.set_trace()
        print config, level
        self.set_race_mode(config['mode'])
        self.nb_lap = config['nb_lap']
        self.cur_lap = config['cur_lap']
        return config

    def guidance_cfg_callback(self, config):
        rospy.loginfo("  Guidance Reconfigure Request:")
        #print config
        rospy.loginfo("    ignoring it")
        
    def periodic(self):
        if self.mode == Node.mode_racing:
            if self.start_finish_sub.viewing_finish():
                start_points, finish_points, dist_to_finish = self.start_finish_sub.get()
                rospy.loginfo('racing and viewing finish {}'.format(dist_to_finish))
                if dist_to_finish < 0.2: # we're close enough to finish
                    if self.cur_lap == self.nb_lap: # we brake
                        self.update_race_mode(Node.mode_finished)
                    else:
                        self.cur_lap += 1 # or we pass to next lap
        elif self.mode == Node.mode_ready:
            tl_red, _, tl_green = self.traffic_light_sub.get()
            if tl_green and not tl_red : # green light and no red, we race
                self.update_race_mode(Node.mode_racing)
                    
    def set_guidance_mode(self, mode):
        rospy.loginfo(' set guidance mode to {}'.format(mode))
        if self.guidance_cfg_client is not None:
            self.guidance_cfg_client.update_configuration({"guidance_mode":mode})
            
        
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

        
def main(args):
  rospy.init_node('trr_race_manager_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
