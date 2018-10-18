import os, logging, yaml, numpy as np, matplotlib.image, matplotlib.pyplot as plt
import PIL.Image, PIL.ImageDraw
import pdb

LOG = logging.getLogger('two_d_guidance.rosmap')

class ROSMap:

    def __init__(self, **kwargs):
        if 'yaml_path' in kwargs:
            self.load_yaml(kwargs['yaml_path'])
        else:
            self.negate = 0
            self.free_thresh =  0.196
            self.occupied_thresh = 0.65
        if 'size_px' in kwargs:
            self.size_px = kwargs['size_px']
            self.img = PIL.Image.new('L', kwargs['size_px'])
            self.image_draw = PIL.ImageDraw.Draw(self.img)
        if 'resolution' in kwargs:
            self.resolution = kwargs['resolution']
        if 'origin' in kwargs:
            self.origin = np.array(kwargs['origin'])
            
    def load_yaml(self, yaml_path):
        LOG.info(' loading map from yaml {}'.format(yaml_path))
        with open(yaml_path, "r") as f:
            _yaml = yaml.load(f)
            self.img_name = _yaml['image']
            map_img_path = os.path.join(os.path.dirname(yaml_path), self.img_name)
            self.img = matplotlib.image.imread(map_img_path)
            self.height, self.width = self.img.shape
            self.resolution = _yaml['resolution']
            self.origin = _yaml['origin']
            self.negate = _yaml['negate']
            self.occupied_thresh = _yaml['occupied_thresh']
            self.free_thresh = _yaml['free_thresh']

    def save(self, map_dir, map_name):
        LOG.info(' saving map to {} {}'.format(map_dir, map_name))
        self.img_name = '{}.png'.format(map_name)
        img_filename = os.path.join(map_dir, self.img_name)
        self.img.save(img_filename)
        self.write_yaml(map_dir, map_name)
        
    def write_yaml(self, map_dir, map_name):
        yaml_output_file = os.path.join(map_dir, '{}.yaml'.format(map_name))
        with open(yaml_output_file, 'w') as stream:
            stream.write('image: {}\n'.format(self.img_name))
            stream.write('resolution: {}\n'.format(self.resolution))
            stream.write('origin: {}\n'.format(np.array2string(self.origin, separator=',')))
            stream.write('negate: {}\n'.format(self.negate))
            stream.write('occupied_thresh: {}\n'.format(self.occupied_thresh))
            stream.write('free_thresh: {}\n'.format(self.free_thresh))
            
    def world_to_pixel(self, p_w):
        p1 = (p_w[:2] - self.origin[:2])/self.resolution 
        px, py = int(np.round(p1[0])), int(np.round(self.size_px[1]-p1[1]-1))
        return px, py

    def draw_line_world(self, p1_w, p2_w, color, width):
        def as3d(_p): return np.array([_p[0], _p[1], 0])
        p1x, p1y = self.world_to_pixel(as3d(p1_w))
        p2x, p2y = self.world_to_pixel(as3d(p2_w))
        self.image_draw.line([(p1x, p1y), (p2x, p2y)], fill=color, width=width)

    def pixel_to_world(self, p_px, alt=0.):
        p_wx, p_wy = (p_px[0]+0.5)*self.resolution, (self.size_px[1]-p_px[1]-1+0.5)*self.resolution
        return self.origin + [p_wx, p_wy, alt]
