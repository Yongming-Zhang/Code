# -*- coding: utf-8 -*-
from collections import defaultdict
from cta_util import *
from seg.seg_util import tensor_resize
import os
import glob
import numpy
from skimage import measure
import nibabel as nib
from scipy.ndimage import binary_erosion
from lib.utils.mio import json_load, load_string_list, save_string_list, mkdir_safe
from gen_cpr_ann_from_cta import gen_cta_psid_json_dict


def load_pid_xjq_dict():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    cta_ann_json_dir = base_dir + 'annotation/'
    cta_json_dict_list = json_load(cta_ann_json_dir + 'task_1265_0-1588272710.json')
    cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    pid_xjq_dict = defaultdict(list)
    psid_list = sorted(cta_psid_json_dicts.keys())
    for psid in psid_list:
        jd = cta_psid_json_dicts[psid]
        for node in jd['nodes']:
            if isinstance(node['descText'][1][0][0]['select'], list):
                continue
            les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
            detect_necessity = g_detect_necessity_dict[node['descText'][1][2][0]['select']]
            segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
            if les_type == 'mca_mb':
                # print(psid, detect_necessity, segments)
                pid_xjq_dict[psid] += segments
    return psid_list, pid_xjq_dict


def get_largest_region(seg):
    labels = measure.label(seg>0)
    props = measure.regionprops(labels)
    area_labels = []
    for prop in props:
        area_labels.append([prop.area, prop.label])
    area_labels.sort()
    top1_area_mask = (labels == area_labels[-1][1])
    if len(area_labels) <= 1:
        top2_area_mask = top1_area_mask
    else:
        top2_area_mask = top1_area_mask + (labels == area_labels[-2][1])
    return top1_area_mask, top2_area_mask


def get_erosion_mask(img_nii, mask, erosion_size, spacing):
    spacing_x, spacing_y, spacing_z = img_nii.header['pixdim'][1:4]
    size_x, size_y, size_z = mask.shape
    resize_shape = int(round(size_x * spacing_x / spacing)), int(round(size_y * spacing_y / spacing)), int(round(size_z * spacing_z / spacing))
    resize_mask = tensor_resize(mask, resize_shape)
    er_resize_mask = binary_erosion(resize_mask, structure=numpy.ones([erosion_size, erosion_size, erosion_size]))
    er_mask = tensor_resize(er_resize_mask, mask.shape)
    return er_mask


def compute_vessel_muscle_inter():
    psid_list, pid_xjq_dict = load_pid_xjq_dict()
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    image_nii_dir = '/brain_data/dataset/ccta/image/'
    vessel_mask_dir = '/brain_data/dataset/ccta/mask_all3/'
    muscle_mask_dir = '/breast_data/cta/cardiac_mask_20200506/'
    erosion_size, spacing = 3, 2.0
    save_suffix = '_s%de%d' % (spacing, erosion_size)
    inter_save_dir = base_dir + 'temp/xjq_inter%s/' % save_suffix
    mkdir_safe(inter_save_dir)
    long_short_map = load_psid_long_short_map()
    xjq_vols, none_vols = [], []
    for psid in psid_list:
        short_psid = long_short_map[psid]
        vessel_path = vessel_mask_dir + short_psid + '.nii.gz'
        muscel_path = muscle_mask_dir + short_psid + '_mask.nii.gz'
        if not (os.path.exists(vessel_path) and os.path.exists(muscel_path)):
            continue
        img_nii = nib.load(image_nii_dir + short_psid + '.nii.gz')
        vessel_mask_nii = nib.load(vessel_path)
        raw_vessel_mask = vessel_mask_nii.get_data()
        muscle_mask = nib.load(muscel_path).get_data()
        vessel_mask = raw_vessel_mask >= 2
        muscle_mask = muscle_mask > 0
        inter_mask = vessel_mask * muscle_mask
        # inter_mask_top1 = get_largest_region(inter_mask)[0]
        xjq = str(pid_xjq_dict[psid]) if psid in pid_xjq_dict else 'None'
        vol_vessel, vol_muscle = vessel_mask.sum(), muscle_mask.sum()
        vol_inter = int(numpy.sum(inter_mask))
        er_muscle_mask = get_erosion_mask(img_nii, muscle_mask, erosion_size, spacing)
        er_inter_mask = vessel_mask * er_muscle_mask
        vol_er_muscle, vol_er_inter = er_muscle_mask.sum(), int(numpy.sum(er_inter_mask))
        # vol_inter_top1 = int(numpy.sum(inter_mask_top1))
        vol_inter_top1 = 0
        vols = [vol_vessel, vol_muscle, vol_inter, vol_inter_top1, vol_er_muscle, vol_er_inter]
        if psid in pid_xjq_dict:
            xjq_vols.append(vols)
        else:
            none_vols.append(vols)
        raw_vessel_mask[er_inter_mask > 0] = 3
        new_vessel_mask_nii = nib.Nifti1Image(raw_vessel_mask, vessel_mask_nii.affine)
        nib.save(new_vessel_mask_nii, inter_save_dir + short_psid + '%s.nii.gz' % save_suffix)
        print('%s: xjq=%s, vol_ves=%d, vol_mus=%d, vol_inter=%d, vol_inter_top1=%d, vol_er_mus=%d, vol_er_inter=%d' % (
            psid, xjq, vol_vessel, vol_muscle, vol_inter, vol_inter_top1, vol_er_muscle, vol_er_inter))
    xjq_vols, none_vols = numpy.float32(xjq_vols), numpy.float32(none_vols)
    print(xjq_vols.mean(axis=0))
    print(none_vols.mean(axis=0))
    numpy.save(base_dir + 'temp/xjq_vols%s.npy' % save_suffix, xjq_vols)
    numpy.save(base_dir + 'temp/none_vols%s.npy' % save_suffix, none_vols)


def gen_long_short_pid_map():
    map_file_list = glob.glob('/brain_data/dataset/ccta/src_dicoms/CCTA_*.txt')
    new_lines = []
    for map_file_path in map_file_list:
        for line in load_string_list(map_file_path):
            short_pid, long_pid = line.split(' ')
            pid, study, series = long_pid.split('/')[-3:]
            new_lines.append(short_pid + ' ' + pid + '/' + study + '/' + series)
    save_string_list('/breast_data/cta/new_data/pss_long_short_map/b1_3580.txt', sorted(new_lines))


def load_psid_long_short_map():
    long_short_map = {}
    for line in load_string_list('/breast_data/cta/new_data/pss_long_short_map/b1_3580.txt'):
        short_psid, long_pss = line.split(' ')
        pid, study, series = long_pss.split('/')
        long_short_map[pid + '_' + study] = short_psid
    return long_short_map


if __name__ == '__main__':
    # gen_long_short_pid_map()
    compute_vessel_muscle_inter()

