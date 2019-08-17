#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
#from geometry_msgs.msg import PolygonStamped, Point32
import pdb

import two_d_guidance.trr.utils as trr_u
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se
import two_d_guidance.cfg.trr_race_managerConfig
import two_d_guidance.srv

# TODO: add traffic light display

class Node:
    mode_staging, mode_ready, mode_racing, mode_finished, mode_join_start = range(5)
    def __init__(self, autostart=False, low_freq=20.):
        self.low_freq = low_freq
        self.cur_lap, self.nb_lap = 0, 1
        # we publish our status
        self.status_pub = trr_rpu.RaceManagerStatusPublisher()
        # we expose a service to be informed when landmarks are passed
        self.lm_service = rospy.Service('LandmarkPassed', two_d_guidance.srv.LandmarkPassed, self.on_landmark_passed)
        self.guidance_cfg_client, self.race_manager_cfg_srv = None, None 
        # we manipulate parameters exposed by the guidance node
        guidance_client_name = "trr_guidance_node"
        self.guidance_cfg_client = dynamic_reconfigure.client.Client(guidance_client_name, timeout=30, config_callback=self.guidance_cfg_callback)
        rospy.loginfo(' guidance_client_name: {}'.format(guidance_client_name))
        # we will expose some parameters to users
        self.race_manager_cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_race_managerConfig, self.race_manager_cfg_callback)
        self.dyn_cfg_update_race_mode(Node.mode_racing if autostart else Node.mode_staging)

        self.state_est_sub = trr_rpu.TrrStateEstimationSubscriber(what='race_manager')
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()


    def on_landmark_passed(self, req):
        print req.id
        if req.id == 1: # FIXME....
            self.cur_lap += 1
            print 'cur_lap'.format(self.cur_lap)
        return two_d_guidance.srv.LandmarkPassedResponse(0)
        
    def dyn_cfg_update_race_mode(self, mode): self.race_manager_cfg_srv.update_configuration({'mode': mode})

    def dyn_cfg_update_cur_lap(self, _v):
        print('dyn_cfg_update_cur_lap {}'.format(_v))
        self.cur_lap = _v
        self.race_manager_cfg_srv.update_configuration({'cur_lap': self.cur_lap})

    def exit(self): self.set_guidance_mode(0) # set guidance is to idle when exiting
        
        # reconfigure (dyn config) race_mode    
    def set_race_mode(self, mode):
        mode_name = ['Staging', 'Ready', 'Racing', 'Finished', 'JoinStart']
        rospy.loginfo('  set race mode to {}'.format(mode_name[mode]))
        self.mode = mode
        cbks = [self.enter_staging, self.enter_ready, self.enter_racing, self.enter_finished, self.enter_join_start]
        cbks[self.mode]()
        
    def set_cur_lap(self, v): self.cur_lap = v

    def race_manager_cfg_callback(self, config, level):
        rospy.loginfo("  Race Manager Reconfigure Request:")
        #pdb.set_trace()
        #print config, level
        self.set_race_mode(config['mode'])
        self.nb_lap = config['nb_lap']
        #self.cur_lap = config['cur_lap']
        return config

    def guidance_cfg_callback(self, config):
        #rospy.loginfo("  Guidance Reconfigure Request:")
        #print config
        #rospy.loginfo("    ignoring it")
        pass
        
    def periodic(self):
        cbks = [self.periodic_staging, self.periodic_ready, self.periodic_racing, self.periodic_finished, self.periodic_join_start]
        try:
            s_est, v_est, cur_lap, dist_to_start, dist_to_finish = self.state_est_sub.get()
            # if self.start_crossed:
            #     rospy.loginfo('race manager periodic: start crossed')
            # if self.finish_crossed:
            #     rospy.loginfo('race manager periodic: finish crossed')
            cbks[self.mode]()
        except trr_rpu.NoRXMsgException:
            rospy.loginfo_throttle(1., 'NoRXMsgException')
        except trr_rpu.RXMsgTimeoutException:
            rospy.loginfo_throttle(1., 'RXMsgTimeoutException')
        self.status_pub.publish(self)
            
    def set_guidance_mode(self, mode):
        mode_name = ['Idle', 'Stop', 'Drive']
        rospy.loginfo('   set guidance mode to {}'.format(mode_name[mode]))
        if self.guidance_cfg_client is not None:
            self.guidance_cfg_client.update_configuration({"guidance_mode":mode})


            
    # Race modes behaviour

    # -Staging: we do nothing
    #
    def enter_staging(self):
        self.set_guidance_mode(0)      # guidance is idle when staging

    def periodic_staging(self): pass   # and we do nothing
    
    # -Ready: we wait for green light
    #
    def enter_ready(self):
        self.set_guidance_mode(1) # guidance mode is stopped

    def periodic_ready(self):
        tl_red, _, tl_green = self.traffic_light_sub.get()
        if tl_green and not tl_red : # green light and no red, we race
            self.dyn_cfg_update_race_mode(Node.mode_racing)

    # -Racing: we check for end of race
    #
    def enter_racing(self):
        rospy.loginfo('    entering racing: lap ({}/{})'.format(self.cur_lap, self.nb_lap))
        #self.dyn_cfg_update_cur_lap(0)
        self.cur_lap = 0
        #self.start_crossed, self.finish_crossed = False, False
        self.set_guidance_mode(2) # guidance is driving when racing

    def periodic_racing(self):
        s_est, v_est, cur_lap, dist_to_start, dist_to_finish = self.state_est_sub.get()
        #rospy.loginfo('racing_periodic: finish crossed {}'.format(self.finish_crossed))
        #if self.finish_crossed:
            #rospy.loginfo('racing_periodic: finish crossed {}/{}'.format(self.cur_lap, self.nb_lap))
        if self.cur_lap >= self.nb_lap: # we brake
            rospy.loginfo('final lap ({}/{}): braking'.format(self.cur_lap, self.nb_lap))
            self.dyn_cfg_update_race_mode(Node.mode_finished)
            #else:
                #self.dyn_cfg_update_cur_lap(self.cur_lap+1) # or we pass to next lap
            #    self.cur_lap += 1
            #    rospy.loginfo('starting lap {}'.format(self.cur_lap))
            #self.finish_crossed = False

    # -Finished: we do nothing
    #
    def enter_finished(self):
        self.set_guidance_mode(1) # guidance mode is stopped

    def periodic_finished(self): pass
                
    # -Join Start: we check for approaching start line 
    #
    def enter_join_start(self):
        self.set_guidance_mode(2) # driving when going to start

    def periodic_join_start(self, dist_to_stop_at=0.15):
        s_est, v_est, cur_lap, dist_to_start, dist_to_finish = self.state_est_sub.get()
        #dist_to_start = self.state_estimator.dist_to_start()
        if dist_to_start < dist_to_stop_at:
            rospy.loginfo('start joined, going to ready')
            self.dyn_cfg_update_race_mode(Node.mode_ready)
        
                

            
        
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
            #self.exit()

        
def main(args):
  rospy.init_node('trr_race_manager_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
