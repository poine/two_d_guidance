#!/usr/bin/env python
import time, numpy as np, matplotlib.pyplot as plt, cv2
import rosbag, cv_bridge
import pdb

import smocap
import two_d_guidance.trr.vision.utils  as trr_vu
import two_d_guidance.trr.vision.lane_1 as trr_l1
import two_d_guidance.trr.vision.lane_2 as trr_l2
import two_d_guidance.trr.vision.lane_3 as trr_l3
import two_d_guidance.trr.vision.lane_4 as trr_l4

def test_on_img(pipe, cam, img):
    pipe.process_image(img, cam, None, None)
    out_img = pipe.draw_debug_bgr(cam)
    cv2.imshow('in', img); cv2.imshow('out', out_img)
    cv2.waitKey(0)


def test_on_bag(pipe, cam, bag_path, img_topic='/camera1/image_raw', sleep=False, talk=False ):
    bag, bridge = rosbag.Bag(bag_path, "r"), cv_bridge.CvBridge()
    durations, last_img_t = [], None
    lane_mod_coefs = []
    for topic, msg, img_t in bag.read_messages(topics=[img_topic]):
        img_dt = 0.01 if last_img_t is None else (img_t-last_img_t).to_sec()
        # this has been settled - we consume bgr image ( same as opencv )
        #cv_img = bridge.imgmsg_to_cv2(msg, "rgb8")
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        pipe.process_image(cv_img, cam, msg.header.stamp, msg.header.seq)
        lane_mod_coefs.append(pipe.lane_model.coefs)
        durations.append(pipe.last_processing_duration)
        if talk: print('{:.3f}s ({:.1f}hz)'.format(pipe.last_processing_duration, 1./pipe.last_processing_duration ))
        if pipe.display_mode != pipe.show_none:
            out_img = pipe.draw_debug_bgr(cam)
            cv2.imshow('out', out_img)
            cv2.waitKey(10)
        last_img_t = img_t
        time_to_sleep = max(0., img_dt-pipe.last_processing_duration) if sleep else 0
        time.sleep(time_to_sleep)
        
    freqs = 1./np.array(durations); _mean, _std, _min, _max = np.mean(freqs), np.std(freqs), np.min(freqs), np.max(freqs)
    plt.hist(freqs); plt.xlabel('hz'); plt.legend(['mean {:.1f} std {:.1f}\n min {:.1f} max {:.1f}'.format(_mean, _std, _min, _max)])

    lane_mod_coefs = np.array(lane_mod_coefs)
    fig, axs = plt.subplots(4, 2)
    for i in range(4):
        axs[i, 0].plot(lane_mod_coefs[:,i])
        axs[i, 1].plot(lane_mod_coefs[:-1,i] - lane_mod_coefs[1:,i])
    plt.show()

    
if __name__ == '__main__':
    
    robot_pierrette, robot_caroline, robot_christine = range(3)
    robot_names = ['pierrette', 'caroline', 'christine']
    robot = robot_christine
    intr_cam_calib_dir = '/home/poine/.ros/camera_info/'
    intr_cam_calib_path = '{}/{}_camera_road_front.yaml'.format(intr_cam_calib_dir, robot_names[robot])
    be_param = trr_vu.NamedBirdEyeParam(robot_names[robot])
    if robot == robot_pierrette:
        extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/pierrette_cam1_extr.yaml'
    elif robot == robot_caroline:
        extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/caroline_cam_road_front_extr.yaml'
    elif robot == robot_christine:
        extr_cam_calib_path = '/home/poine/work/oscar/oscar/oscar_description/cfg/christine_cam_road_front_extr.yaml'
    cam = trr_vu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)

    pipe_1, pipe_2, pipe_3, pipe_4 = range(4)
    pipe_type = pipe_3
    if pipe_type == pipe_1:    # 154hz
        pipe = trr_l1.Contour1Pipeline(cam)
        pipe.thresholder.set_threshold(150)
        pipe.display_mode = pipe.show_contour
    elif pipe_type == pipe_2:  # now 200hz
        pipe = trr_l2.Contour2Pipeline(cam, be_param, use_single_contour=False, ctr_img_min_area=500); # 500
        pipe.use_fancy_filtering = True
        pipe.thresholder.set_threshold(160)  # indoor: 120  outdoor: 160-170
        pipe.set_roi((0, 20), (cam.w, cam.h))
        pipe.display_mode = trr_l2.Contour2Pipeline.show_contour
    elif pipe_type == pipe_3:
        pipe = trr_l3.Contour3Pipeline(cam, be_param)
        pipe.use_fancy_filtering = False
        pipe.thresholder.set_threshold(160)
        pipe.set_roi((0, 20), (cam.w, cam.h))
        pipe.display_mode = pipe.show_contour
    elif pipe_type == pipe_4:
        pipe = trr_l4.Foo4Pipeline(cam, be_param)
        #pipe.thresholder.set_threshold(160)
        pipe.display_mode = pipe.show_lines

    mode_img, mode_bag = range(2)
    mode = mode_bag
    if mode == mode_img:
        #img_path = '/home/poine/work/robot_data/jeanmarie/z_room_line_11.png'
        #img_path = '/home/poine/work/robot_data/caroline/line_z_02.png'
        img_path = '/home/poine/work/robot_data/christine/z_track/image_01.png'
        #img_path = '/home/poine/work/robot_data/christine/vedrines_track/frame_000000.png'
        test_on_img(pipe, cam, cv2.imread(img_path, cv2.IMREAD_COLOR))
    elif mode == mode_bag:
        #bag_path = '/home/poine/2019-07-08-16-37-38.bag' # pierrette
        #bag_path = '/home/poine/2019-07-15-19-01-30.bag'#'/home/poine/2019-07-11-18-08-11.bag' #2019-07-11-15-03-18.bag' # caroline
        #bag_path = '/home/poine/2019-08-30-12-04-21.bag' # caroline vedrines #2019-08-08-16-46-55.bag'
        #bag_path = '/home/poine/2019-09-05-18-30-00.bag' # Christine vedrines failed
        #bag_path = '/home/poine/2019-09-06-12-59-29.bag' # Christine Z failed
        bag_path = '/home/poine/2019-09-10-14-00-00.bag'  # Christine Vedrines OK
        img_topic = '/camera_road_front/image_raw'
        test_on_bag(pipe, cam, bag_path, img_topic, sleep=False, talk=False)
