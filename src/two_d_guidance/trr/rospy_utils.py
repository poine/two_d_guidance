import sys, numpy as np
import cv2, cv_bridge
import rospy, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg, nav_msgs.msg
import pdb

import smocap # for cameras
import smocap.rospy_utils
import two_d_guidance.msg

class OdomListener:
    def __init__(self, odom_topic = '/caroline/diff_drive_controller/odom', callback=None):
        self.odom_sub = rospy.Subscriber(odom_topic, nav_msgs.msg.Odometry, self.callback if callback is None else callback)
        self.lin, self.ang = 0, 0
        
    def callback(self, msg):
        self.msg = msg
        self.lin, self.ang = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        

        
class ImgPublisher:
    def __init__(self, cam, img_topic = "/trr_vision/start_finish/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        
    def publish(self, producer, cam):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(producer.draw_debug(cam), "rgb8"))

class CompressedImgPublisher:
    def __init__(self, cam, img_topic):
        img_topic = img_topic + "/compressed"
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.CompressedImage, queue_size=1)
        
    def publish(self, model, data):
        img_rgb = model.draw_debug(data)
        self.publish2(img_rgb)
        
    def publish2(self, img_rgb):
        img_bgr =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tostring()
        self.image_pub.publish(msg)
        
        
def msgPoint(x, y, z): p = geometry_msgs.msg.Point(); p.x=x; p.y=y;p.z=z; return p


class ContourPublisher:
    def __init__(self, frame_id='caroline/base_link_footprint', topic='/contour',
                 contour_blf=None, rgba=(1.,1.,0.,1.)):
        self.frame_id, self.rgba = frame_id, rgba
        self.pub = rospy.Publisher(topic , visualization_msgs.msg.MarkerArray, queue_size=1)
        rospy.loginfo(' publishing contour on ({})'.format(topic))
        self.msg = visualization_msgs.msg.MarkerArray()
        # if we were given a contour, setup our marker message
        if contour_blf is not None:
            self.msg.markers=[self.get_marker(contour_blf, self.rgba)]

    def get_marker(self, contour_blf, rgba=(1.,1.,0.,1.), closed=True):
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = self.frame_id
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.id = 0
        marker.text = 'foo'
        s = marker.scale; s.x, s.y, s.z = 0.01, 0.2, 0.2
        c = marker.color; c.r, c.g, c.b, c.a = rgba
        o = marker.pose.orientation; o.w, o.x, o.y, o.z = 1, 0, 0, 0
        for x, y, z in contour_blf:
            marker.points.append(msgPoint(x, y, z))
        if closed:
            x, y, z = contour_blf[0]
            marker.points.append(msgPoint(x, y, z))
        return marker

    def publish(self, contour_blf=None):
        if contour_blf is not None:
            self.msg.markers=[self.get_marker(contour_blf, self.rgba)]
        self.pub.publish(self.msg)
        

class TrrStartFinishPublisher:
    def __init__(self, topic='trr_vision/start_finish/status'):
        rospy.loginfo(' publishing start finish status on ({})'.format(topic))
        self.pub = rospy.Publisher(topic, two_d_guidance.msg.TrrStartFinish, queue_size=1)

    def publish(self, pl):
        msg = two_d_guidance.msg.TrrStartFinish()
        if pl.ss_dtc.sees_start():
            for (x, y, z) in pl.start_ctr_lfp:
               msg.start_points.append(msgPoint(x, y, z))
        if pl.ss_dtc.sees_finish():
            for (x, y, z) in pl.finish_ctr_lfp:
               msg.finish_points.append(msgPoint(x, y, z))
        msg.dist_to_finish = pl.dist_to_finish
        msg.dist_to_start = pl.dist_to_start
        self.pub.publish(msg)

        
class TrrStartFinishSubscriber:
    def __init__(self, topic='trr_vision/start_finish/status', user_callback=None):
        self.sub = rospy.Subscriber(topic, two_d_guidance.msg.TrrStartFinish, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed to ({})'.format(topic))
        self.user_callback = user_callback
        self.msg = None
        self.timeout = 0.5
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()
        if self.user_callback: self.user_callback()

    def get(self):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            #pdb.set_trace()
            def array_of_pts(pts): return [[p.x, p.y, p.z] for p in  pts]
            return array_of_pts(self.msg.start_points), array_of_pts(self.msg.finish_points), self.msg.dist_to_start, self.msg.dist_to_finish 
        else:
            return [], [], np.inf, np.inf

    def viewing_finish(self):
        return (self.msg is not None) and\
               ((rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout) and\
               (self.msg.finish_points)
        

class TrrTrafficLightPublisher:
    def __init__(self, topic='trr_vision/traffic_light/status'):
        rospy.loginfo(' publishing traffic light status on ({})'.format(topic))
        self.pub = rospy.Publisher(topic, two_d_guidance.msg.TrrTrafficLight, queue_size=1)
    def publish(self, pl):
        msg = two_d_guidance.msg.TrrTrafficLight()
        msg.red, msg.yellow, msg.green = pl.get_light_status()
        self.pub.publish(msg)

class TrrTrafficLightSubscriber:
    def __init__(self, topic='trr_vision/traffic_light/status'):
        self.sub = rospy.Subscriber(topic, two_d_guidance.msg.TrrTrafficLight, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed to ({})'.format(topic))
        self.msg = None
        self.timeout = 0.5     

    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()

    def get(self):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            return self.msg.red, self.msg.yellow, self.msg.green
        else: return False, False, False


class SimplePublisher:
    def __init__(self, msg_class, topic, what):
        rospy.loginfo(' publishing {} status on {}'.format(what, topic))
        self.pub = rospy.Publisher(topic, msg_class, queue_size=1)
        self.msg_class = msg_class
        
class TrrStateEstimationPublisher(SimplePublisher):
    def __init__(self, topic='trr_state_est/status'):
        SimplePublisher.__init__(self, two_d_guidance.msg.TrrStateEst, topic, 'state estimation')

    def publish(self, model):
        msg = self.msg_class()
        msg.s = model.sn
        self.pub.publish(msg)

class SimpleSubscriber:
    def __init__(self, msg_class, topic, what, timeout=0.5):
        self.sub = rospy.Subscriber(topic, msg_class, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed {} to ({})'.format(what, topic))
        self.timeout = timeout
        self.msg = None
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()


        
class TrrStateEstimationSubscriber(SimpleSubscriber):
    def __init__(self, topic='trr_state_est/status', what='unkown'):
        SimpleSubscriber.__init__(self, two_d_guidance.msg.TrrStateEst, topic, what)

    def get(self):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            return self.msg.s
        else: return float('inf')

        
class TrrSimpleVisionPipeNode:
    def __init__(self, pipeline_class, pipe_cbk=None, low_freq=10):
        self.low_freq = low_freq
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        cam_names = rospy.get_param('~cameras', prefix(robot_name, 'camera_road_front')).split(',')
        self.ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))

        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names, fetch_extrinsics=True, world=self.ref_frame)
        self.cam = self.cam_sys.cameras[0]; self.cam.set_undistortion_param(alpha=1.)

        self.pipeline = pipeline_class(self.cam)

        #def start(self): # TODO FIXME - make sure this can be started later
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)

    # we get a bgr8 image as input
    def on_image(self, img_bgr, (cam_idx, stamp, seq)):
        self.pipeline.process_image(img_bgr, self.cam_sys.cameras[cam_idx], stamp, seq)
        if self.pipe_cbk is not None: self.pipe_cbk()
        
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass



class DebugImgPublisher:
    def __init__(self, topic_src, topic_sink):
        #self.image_pub = rospy.Publisher(topic_sink+"/compressed", sensor_msgs.msg.CompressedImage, queue_size=1)
        self.image_pub = CompressedImgPublisher(cam=None, img_topic=topic_sink)

        self.img, self.compressed_img = None, None
        img_src_topic = topic_src + '/image_raw/compressed'
        self.img_sub = rospy.Subscriber(img_src_topic, sensor_msgs.msg.CompressedImage, self.img_cbk,  queue_size = 1)
        rospy.loginfo(' subscribed to ({})'.format(img_src_topic))

    def img_cbk(self, msg):
        self.compressed_img = np.fromstring(msg.data, np.uint8)

    def publish(self, model, user_data):
        if self.compressed_img is not None:
            self.img_bgr = cv2.imdecode(self.compressed_img, cv2.IMREAD_COLOR)
            self._draw(self.img_bgr, model, user_data)
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            #img_rgb = self.img[...,::-1] # rgb = bgr[...,::-1] OpenCV image to Matplotlib
            self.image_pub.publish2(self.img_rgb)
