import sys, numpy as np
import cv2, cv_bridge
import rospy, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg
import pdb

import smocap # for cameras
import smocap.rospy_utils
import two_d_guidance.msg

class ImgPublisher:
    def __init__(self, cam, img_topic = "/trr_vision/start_finish/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        
    def publish(self, pipe, cam):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(pipe.draw_debug(cam), "rgb8"))

class CompressedImgPublisher:
    def __init__(self, cam, img_topic):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic+"/compressed", sensor_msgs.msg.CompressedImage, queue_size=1)
        
    def publish(self, pipe, cam):
        img_rgb = pipe.draw_debug(cam)
        img_bgr =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tostring()
        self.image_pub.publish(msg)

        
class ContourPublisher:
    def __init__(self, frame_id='caroline/base_link_footprint', topic='/follow_line/detected_contour',
                 contour_blf=None, rgba=(1.,1.,0.,1.)):
        self.frame_id = frame_id
        self.rgba = rgba
        self.pub = rospy.Publisher(topic , visualization_msgs.msg.MarkerArray, queue_size=1)
        rospy.loginfo(' publishing contour on ({})'.format(topic))
        self.msg = visualization_msgs.msg.MarkerArray()
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
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        for x, y, z in contour_blf:
            p1 = geometry_msgs.msg.Point()
            p1.x=x; p1.y=y;p1.z=z
            marker.points.append(p1)
        if closed:
            p = geometry_msgs.msg.Point()
            p.x, p.y, p.z = contour_blf[0]
            marker.points.append(p)
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
        if pl.green_ctr_lfp is not None:
            for (x, y, z) in pl.green_ctr_lfp:
               p = geometry_msgs.msg.Point()
               p.x=x; p.y=y;p.z=z
               msg.start_points.append(p)
        if pl.red_ctr_lfp is not None:
            for (x, y, z) in pl.red_ctr_lfp:
               p = geometry_msgs.msg.Point()
               p.x=x; p.y=y;p.z=z
               msg.finish_points.append(p)
            msg.dist_to_finish = pl.dist_to_finish
                
        self.pub.publish(msg)

        
class TrrStartFinishSubscriber:
    def __init__(self, topic='trr_vision/start_finish/status'):
        self.sub = rospy.Subscriber(topic, two_d_guidance.msg.TrrStartFinish, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed to ({})'.format(topic))
        self.msg = None
        self.timeout = 0.5
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()

    def get(self):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            #pdb.set_trace()
            def array_of_pts(pts): return [[p.x, p.y, p.z] for p in  pts]
            return array_of_pts(self.msg.start_points), array_of_pts(self.msg.finish_points), self.msg.dist_to_finish
        else:
            return [], [], 0.

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
        msg.red = pl.sees_red()
        msg.yellow = False#pl.red_ctr_detc.has_contour()
        msg.green = pl.green_ctr_detc.has_contour()
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
        
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)

    # we get a bgr8 image as input
    def on_image(self, img, (cam_idx, stamp, seq)):
        #img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bgr = img
        #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
