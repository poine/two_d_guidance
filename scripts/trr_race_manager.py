#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server

import pdb

import two_d_guidance.trr.utils as trr_u
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se
import two_d_guidance.cfg.trr_race_managerConfig
import two_d_guidance.srv

import trr.race_manager as trr_rm

# TODO: add traffic light display

class Node(trr_rpu.PeriodicNode):
    mode_staging, mode_ready, mode_racing, mode_finished, mode_join_start = range(5)
    def __init__(self, autostart=False):
        trr_rpu.PeriodicNode.__init__(self, 'race_manager_node')
        self.race_manager = trr_rm.RaceManager()
        # we publish our status
        self.status_pub = trr_rpu.RaceManagerStatusPublisher()
        # we expose a service to be informed ( by state estimation) when landmarks are passed
        self.lm_service = rospy.Service('LandmarkPassed', two_d_guidance.srv.LandmarkPassed, self.on_landmark_passed)
        self.guidance_cfg_client, self.race_manager_cfg_srv = None, None 
        # we manipulate parameters exposed by the guidance node
        guidance_client_name = "trr_guidance_node"
        self.guidance_cfg_client = dynamic_reconfigure.client.Client(guidance_client_name, timeout=30, config_callback=self.guidance_cfg_callback)
        rospy.loginfo(' guidance_client_name: {}'.format(guidance_client_name))
        # we will expose some parameters to users
        self.race_manager_cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_race_managerConfig, self.race_manager_cfg_callback)
        self.dyn_cfg_update_race_mode(Node.mode_racing if autostart else Node.mode_staging)
        # we subscribe to state estimator and traffic light
        self.state_est_sub = trr_rpu.TrrStateEstimationSubscriber(what='race_manager')
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()

    def on_landmark_passed(self, req):
        print('on_landmark_passed {}'.format(req.id))
        if req.id == 1: # FIXME.... path need to be visible
            self.race_manager.next_lap()
        return two_d_guidance.srv.LandmarkPassedResponse(0)
        
    def dyn_cfg_update_race_mode(self, mode):
        self.race_manager_cfg_srv.update_configuration({'mode': mode})

    # remove that for now
    # def dyn_cfg_update_cur_lap(self, _v):
    #     print('dyn_cfg_update_cur_lap {}'.format(_v))
    #     self.race_manager.set_cur_lap(_v)
    #     self.race_manager_cfg_srv.update_configuration({'cur_lap': self.cur_lap})

    def exit(self): self.set_guidance_mode(0) # set guidance is to idle when exiting
        
    # reconfigure (dyn config) race_mode    
    def set_race_mode(self, mode):
        mode_name = ['Staging', 'Ready', 'Racing', 'Finished', 'JoinStart']
        rospy.loginfo('  set race mode to {}'.format(mode_name[mode]))
        self.race_manager.set_mode(mode, self)
        
    #def set_cur_lap(self, v): self.race_manager.set_cur_lap(v)

    def race_manager_cfg_callback(self, config, level):
        rospy.loginfo("  Race Manager Reconfigure Request:")
        #pdb.set_trace()
        #print config, level
        self.set_race_mode(config['mode'])
        self.nb_lap = config['nb_lap']
        #self.cur_lap = config['cur_lap']
        return config

    # I get a confirmation when guidance config is changed
    def guidance_cfg_callback(self, config): pass
        
    def periodic(self):
        try:
            self.race_manager.periodic(self.state_est_sub, self.traffic_light_sub, self.dyn_cfg_update_race_mode)
        except trr_rpu.NoRXMsgException:
            rospy.loginfo_throttle(1., 'NoRXMsgException')
        except trr_rpu.RXMsgTimeoutException:
            rospy.loginfo_throttle(1., 'RXMsgTimeoutException')
        self.status_pub.publish(self.race_manager)
            
    def set_guidance_mode(self, mode):
        mode_name = ['Idle', 'Stop', 'Drive']
        rospy.loginfo('   set guidance mode to {}'.format(mode_name[mode]))
        if self.guidance_cfg_client is not None:
            self.guidance_cfg_client.update_configuration({"guidance_mode":mode})


            
   

        
def main(args):
    Node().run(20)


if __name__ == '__main__':
    main(sys.argv)
