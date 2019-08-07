#!/usr/bin/env python
import argparse, logging, sys, os, math, numpy as np, cv2, gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, GObject
import matplotlib
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import yaml
import pdb

#import smocap
import two_d_guidance.trr_vision_utils as trr_vu

'''
This is a graphical interface for tunning opencv blob detector

smocap/scripts/test_blob_detector.py -i smocap/test/debug_2019_03/roscar_z_01.png -c smocap/params/enac_demo_z/expe_z_detector_default.yaml -e mono8

'''

LOG = logging.getLogger('test_blob_detector')

#http://www.learnopencv.com/blob-detection-using-opencv-python-c/

detector_params_desc = [
    {
        'name':'filterByArea',
        'type':'bool',
        'params': ['minArea', 'maxArea']
    },
    {
        'name':'filterByCircularity',
        'params': ['minCircularity', 'maxCircularity']
    },
    {
        'name':'filterByConvexity',
        'params': ['minConvexity', 'maxConvexity']
    },
    {
        'name':'filterByInertia',
        'params': ['minInertiaRatio', 'maxInertiaRatio']
    },
    {
        'name':'filterByColor',
        'params': ['blobColor']
    },
    'minDistBetweenBlobs',
    'minRepeatability',
    'minThreshold',
    'maxThreshold',
    'thresholdStep'
]


detector_defaults = {}
#    'minArea': 12.,
#    'maxArea': 300.,
#    'blobColor': 255,
#    'minDistBetweenBlobs': 6.
#}




class GUI:
    def __init__(self):
        self.last_dir = os.getcwd()
        self.b = Gtk.Builder()
        gui_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_colored_blob_detector_gui.xml')
        self.b.add_from_file(gui_xml_path)
        self.window = self.b.get_object("window")
        self.window.set_title('BlobDetector')

        self.f = matplotlib.figure.Figure()
        self.ax = self.f.add_subplot(111)
        self.f.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0, wspace=0)

        self.canvas = FigureCanvas(self.f)
        self.b.get_object("alignment_img").add(self.canvas)

        grid = self.b.get_object("grid_params")

        j = 0
        self.detector_params_toggle_buttons = {}
        self.detector_params_entries = {}
        for p in detector_params_desc:
            if isinstance(p, dict):
                #print 'dict', p
                button = Gtk.CheckButton(p['name'])
                grid.attach(button, 0, j, 1, 1)
                self.detector_params_toggle_buttons[p['name']]= button
                j+=1
                for sp in p['params']:
                    label = Gtk.Label('{}'.format(sp))
                    label.set_justify(Gtk.Justification.LEFT)
                    grid.attach(label, 0, j, 1, 1)
                    if 0:
                        adj = Gtk.Adjustment(0, 0, 100, 5, 10, 0)
                        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
                        grid.attach(scale, 1, j, 2, 1)
                    else:
                        entry = Gtk.Entry()
                        grid.attach(entry, 1, j, 2, 1)
                        self.detector_params_entries[sp] = entry
                    #print sp
                    j+=1
            else:
                label = Gtk.Label('{}'.format(p))  
                label.set_justify(Gtk.Justification.LEFT)
                grid.attach(label, 0, j, 1, 1)
                entry = Gtk.Entry()
                grid.attach(entry, 1, j, 2, 1)
                self.detector_params_entries[p] = entry
                j+=1

        scale = self.b.get_object("scale_gamma")
        adj = Gtk.Adjustment(1., 0.1, 2., 0.05, 0.1, 0)
        scale.set_adjustment(adj)

        self.image_display_mode = 'Original'
                
        self.window.show_all()

    def display_image(self, model):
        label = self.b.get_object("label_image")
        label.set_text(model.image_path)
        
        img = model.get_image(self.image_display_mode)
        if len(img.shape) == 2:
            img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img2)
        self.canvas.draw()

    def display_params(self ,params):
        for p in trr_vu.ColoredBlobDetector.param_names:
            a = getattr(params, p)
            if isinstance(a, bool):
                self.detector_params_toggle_buttons[p].set_active(a)
            else:
                self.detector_params_entries[p].set_text(str(a))
                
    def display_detector_res(self, keypoints):
        textview = self.b.get_object("textview1")
        textbuffer = textview.get_buffer()
        txt = "{} keypoints\n\n".format(len(keypoints))
        for kp in keypoints:
            txt += '\tpos: {:.2f} {:.2f}\n'.format(*kp.pt)
            txt += '\tsize: {:.2f}\n'.format(kp.size)
            #txt += '\tangle: {}\n'.format(kp.angle)
            #txt += '\tresp: {}\n'.format(kp.response)
            #txt += '\toctave: {}\n'.format(kp.octave)
            txt += '\n'
        textbuffer.set_text(txt)
            
    def request_path(self, action):
        dialog = Gtk.FileChooserDialog("Please choose a file", self.window, action,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        dialog.set_current_folder(self.last_dir)
        ret = dialog.run()
        file_path = dialog.get_filename() if ret == Gtk.ResponseType.OK else None
        dialog.destroy()
        if file_path is not None:
            self.last_dir = os.path.dirname(file_path)
        return file_path


    def set_image_display_mode(self, model, display_mode):
        self.image_display_mode = display_mode
        self.display_image(model)

class Model:
    def __init__(self, detector_cfg, image_encoding):
        self.gamma = 1.
        self._detector = trr_vu.ColoredBlobDetector(trr_vu.hsv_green_range(), detector_cfg)
        self.image_path = None
        
    def update_param(self, name, value):
        LOG.info(' updating detector param: {} {}'.format(name, value))
        self._detector.update_param(name, value)
 
    def load_image(self, path):
        self.image_path = path
        self.img = cv2.imread(path)

    def correct_gamma(self, gamma=1.):
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
		          for i in np.arange(0, 256)]).astype("uint8")
        self.gamma_corrected_img = cv2.LUT(self.img, table)
        
    def detect_blobs(self):
        self.hsv_img = cv2.cvtColor(self.gamma_corrected_img, cv2.COLOR_BGR2HSV)
        self.keypoints, self.img_coords = self._detector.process_hsv_image(self.hsv_img)
        self.img_res = cv2.drawKeypoints(self.gamma_corrected_img, self.keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def cluster_blobs(self):
        self.clusters_id = self._detector.cluster_keypoints(self.img_coords)
        #print self.clusters
        #self._detector.identify_marker(self.img_coords, self.clusters)
        nb_clusters = np.max(self.clusters_id) + 1
        self.clusters = [self.img_coords[self.clusters_id == i] for i in range(nb_clusters)]
        return self.clusters
        
    def save(self, path):
        LOG.info(' saving detector config to: {}'.format(path))
        self._detector.save_cfg(path)

    def load(self, path):
        LOG.info(' loading detector config from: {}'.format(path))
        self._detector.load_cfg(path)


    def get_image(self, which):
        if which == 'Original':
            return self.img
        elif which == 'Mask':
            return self._detector.mask
        elif which == 'Result':
            return self.img_res
        
        
class App:

    def __init__(self, detector_cfg, image_path, image_encoding):
        self.gui = GUI()
        self.model = Model(detector_cfg, image_encoding)
        self.gui.display_params(self.model._detector.blob_params)
        self.register_gui()
        self.load_image(image_path)

    def register_gui(self):
        self.gui.window.connect("delete-event", self.quit)
        self.gui.b.get_object("button_load_image").connect("clicked", self.on_load_image_clicked)
        self.gui.b.get_object("button_detect").connect("clicked", self.on_detect_clicked)
        self.gui.b.get_object("button_save_params").connect("clicked", self.on_save_params)
        self.gui.b.get_object("button_load_params").connect("clicked", self.on_load_params)
        for name in self.gui.detector_params_toggle_buttons:
            self.gui.detector_params_toggle_buttons[name].connect("toggled", self.on_button_toggled, name)
        for name in self.gui.detector_params_entries:   
            self.gui.detector_params_entries[name].connect("activate", self.on_entry_param_changed, name)
        self.gui.b.get_object('scale_gamma').connect("value-changed", self.on_gamma_changed)
        self.gui.b.get_object('comboboxtext_image').connect("changed", self.on_display_type_changed)

        
    def on_load_image_clicked(self, b):
        path = self.gui.request_path(Gtk.FileChooserAction.OPEN)
        if path is not None: self.load_image(path)

    def on_detect_clicked(self, button):
        self.run_detector()

    def on_save_params(self, button):
        path = self.gui.request_path(Gtk.FileChooserAction.SAVE)
        if path is not None:
            self.model.save(path)

    def on_load_params(self, button):
        path = self.gui.request_path(Gtk.FileChooserAction.OPEN)
        if path is not None:
            self.model.load(path)
            self.gui.display_params(self.model._detector.params)

    def on_button_toggled(self, button, name):
        self.model.update_param(name, button.get_active())

    def on_entry_param_changed(self, entry, name):
        self.model.update_param(name, float(entry.get_text()))

    def on_gamma_changed(self, event):
        self.model.correct_gamma(self.gui.b.get_object('scale_gamma').get_value())

    def on_display_type_changed(self, combo):
        self.gui.set_image_display_mode(self.model, combo.get_active_text())
        
    def load_image(self, path):
        LOG.info(' loading image: {}'.format(path))
        self.model.load_image(path)
        self.model.correct_gamma()
        self.model.detect_blobs()
        self.run_detector()
        
    def run_detector(self):
        self.model.detect_blobs()
        self.gui.display_detector_res(self.model.keypoints)
        #self.gui.display_image(self.model.img_res)
        self.gui.display_image(self.model)
        
        
    def run(self):
        Gtk.main()

    def quit(self, a, b):
        Gtk.main_quit() 




        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=3, linewidth=300)
    parser = argparse.ArgumentParser(description='Tune Blob Detector.')
    parser.add_argument('-i', '--img', default='/home/poine/work/smocap/smocap/test/gazebo_samples/image_09.png')
    parser.add_argument('-c', '--cfg', default='/home/poine/work/smocap/smocap/params/gazebo_detector_cfg.yaml')
    parser.add_argument('-e', '--enc', default='bgr8')
    args = parser.parse_args()
    #pdb.set_trace()
    
    if 0:
        args = {
            'detector_cfg':'/home/poine/work/smocap.git/smocap/params/gazebo_detector_cfg.yaml',
            'image_path':'/home/poine/work/smocap.git/smocap/test/gazebo_samples/image_09.png',
            'image_encoding':'bgr8'
        }
    #App(**args).run()
    App(**{'detector_cfg':args.cfg, 'image_path':args.img, 'image_encoding':args.enc}).run()









    
