#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client,  dynamic_reconfigure.server
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
#from geometry_msgs.msg import PolygonStamped, Point32
import pdb

import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.state_estimation as trr_se
import two_d_guidance.cfg.trr_race_managerConfig


class Node:
    mode_staging, mode_ready, mode_racing, mode_finished, mode_join_start = range(5)
    def __init__(self, autostart=False, low_freq=20.):
        self.low_freq = low_freq
        # for now we run the state estimation
        self.state_estimator = trr_se.StateEstimator(self.lm_passed_cbk)
        self.state_est_pub = trr_rpu.TrrStateEstimationPublisher()
        
        self.crossed_finish = False

        self.guidance_cfg_client, self.race_manager_cfg_srv = None, None 
        # we manipulate parameters exposed by the guidance node
        guidance_client_name = "trr_guidance_node"
        self.guidance_cfg_client = dynamic_reconfigure.client.Client(guidance_client_name, timeout=30, config_callback=self.guidance_cfg_callback)
        rospy.loginfo(' guidance_client_name: {}'.format(guidance_client_name))
        # we will expose some parameters to users
        self.race_manager_cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_race_managerConfig, self.race_manager_cfg_callback)
        self.update_race_mode(Node.mode_racing if autostart else Node.mode_staging)

        self.start_finish_sub = trr_rpu.TrrStartFinishSubscriber(user_callback=self.on_start_finish)
        self.traffic_light_sub = trr_rpu.TrrTrafficLightSubscriber()
        self.odom_sub = trr_rpu.OdomListener(odom_topic='/caroline_robot_hardware/diff_drive_controller/odom', callback=self.on_odom)


    def on_odom(self, msg):
        #print msg.header.seq, msg.header.stamp
        #print msg.twist.twist.linear
        seq, stamp, vx, vy = msg.header.seq, msg.header.stamp, msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.state_estimator.update_odom(seq, stamp, vx, vy)

    def on_start_finish(self):
        start_points, finish_points, dist_to_start, dist_to_finish = self.start_finish_sub.get()
        self.state_estimator.update_landmark(dist_to_start, dist_to_finish)

    def lm_passed_cbk(self, lm_id):
        rospy.loginfo('### lm_passed_cbk: passed {}'.format(self.state_estimator.lm_names[lm_id]))
        if lm_id == self.state_estimator.LM_FINISH: self.finish_crossed = True
        elif lm_id == self.state_estimator.LM_START: self.start_crossed = True
        
    def update_race_mode(self, mode):
        self.race_manager_cfg_srv.update_configuration({'mode': mode})
        
    def enter_staging(self):
        self.set_guidance_mode(0) # idle when staging

    def enter_racing(self):
        self.cur_lap = 0
        self.start_crossed, self.finish_crossed = False, False
        self.set_guidance_mode(2) # driving when racing

    def enter_join_start(self):
        self.set_guidance_mode(2) # driving when going to start


    def periodic_racing(self):
        if self.start_finish_sub.viewing_finish():
            if self.finish_crossed:
                rospy.loginfo('periodic: finish crossed')
                if self.cur_lap == self.nb_lap: # we brake
                    rospy.loginfo('final lap ({}): braking'.format(self.cur_lap))
                    self.update_race_mode(Node.mode_finished)
                else:
                    self.cur_lap += 1 # or we pass to next lap
                    rospy.loginfo('starting lap {}'.format(self.cur_lap))
                self.finish_crossed = False

    def periodic_join_start(self, dist_to_stop_at=0.15):
        dist_to_start = self.state_estimator.dist_to_start()
        if dist_to_start < dist_to_stop_at:
            print('start joined, going to ready')
            self.update_race_mode(Node.mode_ready)
        
    def periodic_ready(self):
        tl_red, _, tl_green = self.traffic_light_sub.get()
        if tl_green and not tl_red : # green light and no red, we race
            self.update_race_mode(Node.mode_racing)

                
    # reconfigure (dyn config) race_mode    
    def set_race_mode(self, mode):
        mode_name = ['Staging', 'Ready', 'Racing', 'Finished', 'JoinStart']
        rospy.loginfo('  set race mode to {}'.format(mode_name[mode]))
        self.mode = mode
        if mode == Node.mode_racing:
            self.enter_racing()
        elif mode == Node.mode_staging:
            self.enter_staging()
        elif mode == Node.mode_ready or mode == Node.mode_finished:
            self.set_guidance_mode(1) # stoped otherwise
        elif mode == Node.mode_join_start:
            self.enter_join_start()

        
    def race_manager_cfg_callback(self, config, level):
        rospy.loginfo("  Race Manager Reconfigure Request:")
        #pdb.set_trace()
        #print config, level
        self.set_race_mode(config['mode'])
        self.nb_lap = config['nb_lap']
        self.cur_lap = config['cur_lap']
        return config

    def guidance_cfg_callback(self, config):
        #rospy.loginfo("  Guidance Reconfigure Request:")
        #print config
        #rospy.loginfo("    ignoring it")
        pass
        
    def periodic(self):

        if self.mode == Node.mode_racing:
            self.periodic_racing()
        if self.mode == Node.mode_join_start:
            self.periodic_join_start()
        elif self.mode == Node.mode_ready:
            self.periodic_ready()

        self.state_est_pub.publish(self.state_estimator)
        #print(self.state_estimator.status())
                    
    def set_guidance_mode(self, mode):
        mode_name = ['Idle', 'Stop', 'Drive']
        rospy.loginfo('   set guidance mode to {}'.format(mode_name[mode]))
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
