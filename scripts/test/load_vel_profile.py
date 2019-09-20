#!/usr/bin/env python

import os, sys, roslib, rospy, rospkg, rostopic
import two_d_guidance.srv

'''
call guidance load_velocity_profile service
'''

def main():
    path_filename = '/home/poine/work/two_d_guidance/paths/demo_z/track_trr_sim_2.npz'
    if len(sys.argv) > 1: path_filename = sys.argv[1]
    print('loading {}'.format(path_filename))
    rospy.init_node('load_vel_profile')
    srv_topic = 'GuidanceLoadVelProf'
    print('Waiting for service: {}'.format(srv_topic))
    rospy.wait_for_service(srv_topic)
    print(' -available')
    _srv_proxy = rospy.ServiceProxy(srv_topic, two_d_guidance.srv.GuidanceLoadVelProf)
    try:
        resp1 = _srv_proxy(path_filename)
    except rospy.ServiceException, e:
        print("Service call failed: {}".format(e))
    print('  -called')

if __name__ == '__main__':
    main()
