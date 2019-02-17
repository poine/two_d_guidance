#!/usr/bin/env python
import time, math, numpy as np
import rospy, geometry_msgs.msg

#import homere_control.msg, homere_control.utils as hcu
import pdb

def step(t, a0=-1, a1=1, dt=4, t0=0): return a0 if math.fmod(t+t0, dt) > dt/2 else a1

def stairs(t,  n_stair=10, dt_stair=0.5, _min=0, _max=20):
    a = int(math.fmod(t, dt_stair*n_stair)/dt_stair)
    return _min + (_max - _min) * a / n_stair


class TwistSine:
    def __init__(self, al=0.75, oml=0.25, aa=0.75, oma=0.25, _type=0):
        self.al, self.aa, self.oml, self.oma = al, oml, aa, oma
        self._type = _type
        
    def to_msg(self, t, msg):
        msg.linear.x = self.al*math.sin(self.oml*t) if self._type in [0,2] else 0.
        msg.angular.z = self.aa*math.sin(self.oma*t) if self._type in [1,2] else 0.
        

class TwistLinSine:
    def __init__(self, a=0.75, om=0.25):
        self.a, self.om = a, om
    def to_msg(self, t, msg):
        msg.linear.x = self.a*math.sin(self.om*t)
        msg.angular.z = 0
        
class TwistAngSine:
    def __init__(self, a=0.75, om=0.25):
        self.a, self.om = a, om
    def to_msg(self, t, msg):
        #pdb.set_trace()
        msg.linear.x = 0
        msg.angular.z = self.a*math.sin(self.om*t)        

class TwistSine2:
    def __init__(self, a1=0.75, om1=0.25, a2=0.75, om2=0.2):
        self.a1, self.om1 = a1, om1
        self.a2, self.om2 = a2, om2
    def to_msg(self, t, msg):
        msg.linear.x = self.a1*math.sin(self.om1*t)
        msg.angular.z = self.a2*math.sin(self.om2*t)
        
# class TwistStep:
#     def __init__(self, a0=-0.1, a1=0.1, dt=10., t0=0):
#         self.a0, self.a1, self.dt, self.t0 = a0, a1, dt, t0

#     def to_msg(self, t, msg):
#         msg.linear.x = step(t, self.a0, self.a1, self.dt, self.t0)
#         msg.angular.z = 0

class TwistLinStep:
    def __init__(self, a0=-0.1, a1=0.1, dt=10., t0=0):
        self.a0, self.a1, self.dt, self.t0 = a0, a1, dt, t0

    def to_msg(self, t, msg):
        msg.linear.x = step(t, self.a0, self.a1, self.dt, self.t0)
        msg.angular.z = 0

class TwistAngStep:
    def __init__(self, a0=-1., a1=1., dt=10., t0=0):
        self.a0, self.a1, self.dt, self.t0 = a0, a1, dt, t0

    def to_msg(self, t, msg):
        msg.linear.x = 0
        msg.angular.z = step(t, self.a0, self.a1, self.dt, self.t0)



        
class TwistRandom:
    def __init__(self, t0, zero_v=False, zero_psid=False, alternate=False, v_amp=1., psid_amp=1.):
        self._min_dur, self._max_dur = 1., 10.
        self.zero_v, self.zero_psid, self.alternate = zero_v, zero_psid, alternate
        self.v_amp, self.psid_amp = v_amp, psid_amp
        self.end_stage = t0
        self.type_stage = 0
        
    def to_msg(self, t, msg):
        if t  >= self.end_stage:
            self.type_stage = (self.type_stage+1)%2
            self.v    = 0 if self.zero_v    or self.alternate and self.type_stage     else np.random.uniform(low=-self.v_amp, high=self.v_amp)
            self.psid = 0 if self.zero_psid or self.alternate and not self.type_stage else np.random.uniform(low=-self.psid_amp, high=self.psid_amp)
            self.end_stage += np.random.uniform(low=self._min_dur, high=self._max_dur)
        msg.linear.x = self.v
        msg.angular.z = self.psid
    
def run_twist():
    rospy.init_node('homere_controller_input_sender', anonymous=True)
    cmd_topic = rospy.get_param('~cmd_topic', '/homere/homere_controller/cmd_vel')
    signal_type = rospy.get_param('~signal_type', 'step_lin')
    duration = rospy.get_param('~duration', 60)
    period  = rospy.get_param('~period', 2.)
    amp  = rospy.get_param('~amp', 0.2)
    rospy.loginfo('  Sending Twist messages to topic {}'.format(cmd_topic))
    rospy.loginfo('  Sending {} for {:.1f} s'.format(signal_type, duration))
    rospy.loginfo('  amp {} period {:.1f} s'.format(amp, period))
    ctl_input_pub = rospy.Publisher(cmd_topic, geometry_msgs.msg.Twist, queue_size=1)
    ctl_in_msg = geometry_msgs.msg.Twist()
    ctl_ins = {'step_lin': lambda: TwistLinStep(a0=-amp, a1=amp, dt=period),
               'step_ang': lambda: TwistAngStep(a0=-amp, a1=amp, dt=period),
               'sine_lin': lambda: TwistLinSine(a=amp, om=2*np.pi/period),
               'sine_ang': lambda: TwistAngSine(a=amp, om=2*np.pi/period),
               'sine_2':   lambda: TwistSine(al=amp, oml=2*np.pi/period, aa=.9, oma=1., _type=2),
               'random_lin': lambda: TwistRandom(t0=rospy.Time.now().to_sec(), zero_psid=True, v_amp=0.2),
               'random_ang': lambda: TwistRandom(t0=rospy.Time.now().to_sec(), zero_v=True),
               'random_alt': lambda: TwistRandom(t0=rospy.Time.now().to_sec(), alternate=True, v_amp=0.2),
               'random_2': lambda: TwistRandom(t0=rospy.Time.now().to_sec(), v_amp=0.2) }
    ctl_in = ctl_ins[signal_type]()
    rate = rospy.Rate(50.)
    rospy.sleep(0.1)
    start = now = rospy.Time.now(); end = start + rospy.Duration(duration)
    #pdb.set_trace()
    i = 0
    while not rospy.is_shutdown() and now < end:
        now = rospy.Time.now()
        elapsed = now - start
        ctl_in.to_msg(now.to_sec(), ctl_in_msg)
        ctl_input_pub.publish(ctl_in_msg)
        if i%10 == 0: print('{:04.1f} s: lin: {:.2f} ang: {:.2f}'.format(elapsed.to_sec(), ctl_in_msg.linear.x, ctl_in_msg.angular.z))
        i += 1
        rate.sleep()

# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 world odom 10        

if __name__ == '__main__':
    try:
        run_twist()
    except rospy.ROSInterruptException: pass
