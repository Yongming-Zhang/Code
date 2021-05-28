import sys
import glob
import cv2
import math
import numpy
import SimpleITK as sitk
sys.path.insert(0, '../../')
from cta.cta_util import gen_image_from_dcm
from lib.image.image import image_center_crop


g_label_dict = {
    'fp': 0,
    'low': 1,
    'cal': 2,
    'mix': 3,
    'stent': 4
}


def load_lumen_as_dict(lumen_dir, center_crop=-1, rescale=-1, lumen_windows=None):
    lumen_paths = sorted(glob.glob(lumen_dir + '*.dcm'))
    d = {}
    if lumen_windows is None:
        lumen_windows = [(-200, 800)]
    for lumen_path in lumen_paths:
        instance_num = int(lumen_path.split('/')[-1][:-4])
        img = gen_image_from_dcm(lumen_path, lumen_windows)[0]
        if center_crop > 0:
            img = image_center_crop(img, center_crop, center_crop)
        if rescale > 0:
            img = cv2.resize(img, (rescale, rescale))
        d[instance_num] = img
    return d


def load_lumen_int16_as_dict(lumen_dir, center_crop=-1):
    lumen_paths = sorted(glob.glob(lumen_dir + '*.dcm'))
    d = {}
    for lumen_path in lumen_paths:
        instance_num = int(lumen_path.split('/')[-1][:-4])
        img16 = numpy.float32(numpy.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(lumen_path)))))
        if center_crop > 0:
            img16 = image_center_crop(img16, center_crop, center_crop)
        d[instance_num] = img16
    return d


def get_centerline3d_lumen_indices(c3d_pts, spacing_x, spacing_y, spacing_z, start_i=0):
    c3d_dist, lumen_i, lumen_indices = 0, start_i, [start_i] * len(c3d_pts)
    for i in range(1, len(c3d_pts)):
        (x0, y0, z0), (x1, y1, z1) = c3d_pts[i - 1], c3d_pts[i]
        c3d_dist += math.sqrt((spacing_x * (x1 - x0)) ** 2 + (spacing_y * (y1 - y0)) ** 2 + (spacing_z * (z1 - z0)) ** 2)
        if c3d_dist >= spacing_x:
            c3d_dist = 0
            lumen_i += 1
        lumen_indices[i] = lumen_i
    return lumen_indices

