#!/usr/bin/env python
import os, time, numpy as np, matplotlib.pyplot as plt, cv2
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


def test_on_bag(pipe, cam, bag_path, img_topic, odom_topic, sleep=False, talk=False, display=True):
    bag, bridge = rosbag.Bag(bag_path, "r"), cv_bridge.CvBridge()
    durations, last_img_t = [], None
    times, lane_mod_coefs = [], []
    odom_times, odom_vlins, odom_vangs = [], [], []
    img_idx = -1
    for topic, msg, img_t in bag.read_messages(topics=[img_topic, odom_topic]):
        if topic == img_topic:
            img_idx += 1
            img_dt = 0.01 if last_img_t is None else (img_t-last_img_t).to_sec()
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            pipe.process_image(cv_img, cam, msg.header.stamp, msg.header.seq)
            lane_mod_coefs.append(pipe.lane_model.coefs)
            times.append(img_t.to_sec())
            durations.append(pipe.last_processing_duration)
            if talk: print('img {} {:.3f}s ({:.1f}hz)'.format(img_idx, pipe.last_processing_duration, 1./pipe.last_processing_duration ))
            if pipe.display_mode != pipe.show_none:
                out_img = pipe.draw_debug_bgr(cam)
                cv2.imshow('input', cv_img)
                cv2.imshow('pipe debug', out_img)
                cv2.waitKey(10)
                #cv2.waitKey(0)
            last_img_t = img_t
            time_to_sleep = max(0., img_dt-pipe.last_processing_duration) if sleep else 0
            time.sleep(time_to_sleep)
        elif topic == odom_topic:
            odom_times.append(msg.header.stamp.to_sec())
            odom_vlins.append(msg.twist.twist.linear.x)
            odom_vangs.append(msg.twist.twist.angular.z)
        
    freqs = 1./np.array(durations); _mean, _std, _min, _max = np.mean(freqs), np.std(freqs), np.min(freqs), np.max(freqs)
    plt.hist(freqs); plt.xlabel('hz'); plt.legend(['mean {:.1f} std {:.1f}\n min {:.1f} max {:.1f}'.format(_mean, _std, _min, _max)])

    lane_mod_coefs = np.array(lane_mod_coefs)
    times = np.array(times)

    filename = '/tmp/pipe_run.npz'
    print('saving run to {}'.format(filename))
    np.savez(filename, times = times, lane_mod_coefs=lane_mod_coefs,
             odom_times=odom_times, odom_vlins=odom_vlins, odom_vangs=odom_vangs)

    plot_run(times, lane_mod_coefs)

    
    
def plot_run(times, lane_mod_coefs):
    fig, axs = plt.subplots(4, 2)
    for i in range(4):
        axs[i, 0].plot(lane_mod_coefs[:,i])
        axs[i, 1].plot(lane_mod_coefs[:-1,i] - lane_mod_coefs[1:,i])
    plt.show()

def test_load(filename='/tmp/pipe_run.npz'):
    print('loading path from {}'.format(filename))
    data =  np.load(filename)
    times, lane_mod_coefs = data['times'], data['lane_mod_coefs']
    plot_run(times, lane_mod_coefs)
    
    
if __name__ == '__main__':
    #test_load()
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
        pipe = trr_l2.Contour2Pipeline(cam, robot_names[robot], use_single_contour=False, ctr_img_min_area=500); # 500
        pipe.use_fancy_filtering = True
        pipe.thresholder.set_threshold(160)  # indoor: 120  outdoor: 160-170
        pipe.set_roi((0, 20), (cam.w, cam.h))
        pipe.display_mode = trr_l2.Contour2Pipeline.show_contour
    elif pipe_type == pipe_3:
        pipe = trr_l3.Contour3Pipeline(cam, robot_names[robot])
        pipe.use_fancy_filtering = False
        pipe.thresholder.set_threshold(160)
        pipe.set_roi((0, 20), (cam.w, cam.h))
        pipe.display_mode = pipe.show_be
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
        bag_dir = '/home/poine'
        # bag_filename = '2019-07-08-16-37-38.bag' # pierrette
        # bag_filename = '2019-07-11-15-03-18.bag' # caroline
        # bag_filename = '2019-07-11-18-08-11.bag' # caroline
        # bag_filename = '2019-07-15-19-01-30.bag' # caroline
        # bag_filename = '2019-08-08-16-46-55.bag' # caroline vedrines
        # bag_filename = '2019-08-30-12-04-21.bag' # caroline vedrines
        # bag_filename = '2019-09-05-18-30-00.bag' # Christine vedrines failed
        # bag_filename = '2019-09-06-12-59-29.bag' # Christine Z failed
        # bag_filename = '2019-09-09-19-08-27.bag' # Christine Z
        # bag_filename = '2019-09-10-14-00-00.bag' # Christine Vedrines 3 tours, nuageux
        # Camera road + odom
        #bag_filename = '2019-09-12-13-09-59.bag' # vedrines 1 tour, soleil haut, 1 ombre poteau, auto gain/exp
        # bag_filename = '2019-09-12-13-12-23.bag' # idem, fail sur ombre, non auto gain, auto exp
        bag_filename = '2019-09-12-13-14-57.bag' # idem, semi fail sur ombre, auto gain, exp mini
        # bag_filename = '2019-09-12-13-16-55.bag' # idem, arret sur ombre, gain bleu a 0
        # bag_filename = '2019-09-12-13-19-53.bag' # idem, ligne droite seule, vitesse 4
        # Camera road + odom + imu
        #bag_filename = '2019-09-16-18-40-00.bag' # Christine Vedrines 1 tour, ombres longues, vitesse constante 2.5
        #bag_filename = '2019-09-16-18-43-00.bag' # Christine Vedrines, ombres longues, acceleration en lignes droites a 4 puis 5
        img_topic, odom_topic = '/camera_road_front/image_raw', '/oscar_ackermann_controller/odom'
        bag_path = os.path.join(bag_dir, bag_filename)
        test_on_bag(pipe, cam, bag_path, img_topic, odom_topic, sleep=False, talk=False)
