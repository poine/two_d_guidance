#!/usr/bin/env python

import sys
import rospy
import two_d_guidance.srv

def set_mode(x, y):
    rospy.wait_for_service('set_mode')
    try:
        srv_proxy = rospy.ServiceProxy('set_mode', two_d_guidance.srv.SetMode)
        resp1 = srv_proxy(x, y)
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "{} [x y]".format(sys.argv[0])

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    else:
        print usage()
        sys.exit(1)
    print "Requesting %s+%s"%(x, y)
    print "%s + %s = %s"%(x, y, set_mode(x, y))
