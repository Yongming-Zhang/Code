# -*- coding: utf-8 -*-
from seg_util import *
import cv2
import glob
import numpy as np
import nibabel as nib
import shutil
from collections import defaultdict
from lib.utils.mio import *


def load_json_dicts():
    lines = load_string_list('/breast_data/cta/new_data/pss_long_short_map/sg_b1_2036.txt')
    pss_map = {}
    for line in lines:
        short_id, long_id = line.split(' ')
        pss_map[long_id] = short_id
    plaque_jd_list = json_load('/breast_data/cta/new_data/b2_sg_407/annotation/task_1265_0-1588272710.json')
    cta_jd_dict, cta_ps_dict = defaultdict(list), {}
    for jd in plaque_jd_list:
        long_id = jd['patientID'] + '/' + jd['studyUID'] + '/' + jd['seriesUID']
        if jd['other_info']['passReason'] != '':
            continue
        if long_id in pss_map:
            short_id = pss_map[long_id]
            cta_jd_dict[short_id] += jd['nodes']
            cta_ps_dict[short_id] = jd['patientID'] + '_' + jd['studyUID']
    print(len(cta_jd_dict))
    return cta_jd_dict, cta_ps_dict


def get_img_mask(subid):
    base_dir = '/home/data/ccta/'
    # base_dir = '/brain_data/dataset/ccta/'
    if not os.path.exists(base_dir + 'mask_all3/' + subid + '.nii'):
        return None, None, None
    mask_nii = nib.load(base_dir + 'mask_all3/' + subid + '.nii')
    mask_ = mask_nii.get_data()
    mask_ = np.transpose(mask_, (2, 1, 0))
    img_nii = nib.load(base_dir + 'image/' + subid + '.nii')
    image_ = img_nii.get_data()
    image_ = np.transpose(image_, (2, 1, 0))
    return image_, mask_, mask_nii.affine


def gen_plaque_mask(mask_shape, nodes, instance_dict):
    mask = np.zeros(mask_shape, dtype='uint8')
    for node in nodes:
        les_type = node['descText'][1][0][0]['select']
        if isinstance(les_type, list):
            continue
        if les_type not in [u'钙化', u'低密度', u'混合', u'闭塞']:
            continue
        for roi in node['rois'] + node['bounds']:
            contours_numpy = np.int32(roi['edge'])
            z = instance_dict[roi['slice_index']]
            cv2.drawContours(mask[z], (contours_numpy,), 0, color=3, thickness=-1)
    return mask


def proc_one(sub_id, nodes, psid):
    cta_dcm_dir = '/breast_data/cta/new_data/b2_sg_407/cta_dicom/'
    paths = sorted(glob.glob(cta_dcm_dir + psid + '/*.dcm'))
    instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(paths)}
    image, vessel_mask, affine = get_img_mask(sub_id)
    if image is None:
        return None, None, None, None
    plaque_mask = gen_plaque_mask(vessel_mask.shape, nodes, instance_dict)
    vnum, pnum = np.sum(vessel_mask > 0), np.sum(plaque_mask > 0)
    s = '%s: v=%d, p=%d' % (sub_id, vnum, pnum)
    vessel_mask[plaque_mask > 0] = 3
    l1, l2, l3 = np.sum(vessel_mask == 1), np.sum(vessel_mask == 2), np.sum(vessel_mask == 3)
    vp_num = vnum + pnum - np.sum(vessel_mask > 0)
    s += ', v&p=%d, l1=%d, l2=%d, l3=%d' % (vp_num, l1, l2, l3)
    return np.transpose(vessel_mask, (2, 1, 0)), affine, np.int32([l1, l2, l3]), s


def proc_main():
    cta_jd_dict, cta_psid_dict = load_json_dicts()
    sub_id_list = sorted(cta_jd_dict.keys())
    total_pixel_num = np.int32([0, 0, 0])
    save_dir = '/home/data/ccta/mask_wp4/'
    mkdir_safe(save_dir)
    for i, sub_id in enumerate(sub_id_list):
        if i <= 121:
            continue
        mask, affine, pixel_num, s = proc_one(sub_id, cta_jd_dict[sub_id], cta_psid_dict[sub_id])
        if mask is None:
            continue
        total_pixel_num += pixel_num
        print('No.%d/%d, %s, total: %s' % (i, len(sub_id_list), s, str(total_pixel_num)))
        nib.save(nib.Nifti1Image(mask, affine), save_dir + '%s.nii' % sub_id)
        shutil.copy('/brain_data/dataset/ccta/heatmap/%s.nii.gz' % sub_id, '/home/data/ccta/heatmap/%s.nii.gz' % sub_id)


def load_lines_as_dict(path):
    return {line: True for line in load_string_list(path)}


def split_list():
    mask_dir = '/home/data/ccta/mask_wp4/'
    sub_id_list = [p.split('/')[-1][:-4] for p in sorted(glob.glob(mask_dir + '*.nii'))]
    train_sub_dict = load_lines_as_dict('/brain_data/dataset/ccta/train_all_subjects_20200425.lst')
    valid_sub_dict = load_lines_as_dict('/brain_data/dataset/ccta/validate_all_subjects_20200410.lst')
    train_list, valid_list, test_list = [], [], []
    for sub_id in sub_id_list:
        if sub_id in train_sub_dict:
            train_list.append(sub_id)
        elif sub_id in valid_sub_dict:
            valid_list.append(sub_id)
        else:
            test_list.append(sub_id)
    save_dir = '/breast_data/cta/new_data/vessel_seg/with_sg_407_plaque/config/'
    save_string_list(save_dir + 'train_list.txt', sorted(train_list))
    save_string_list(save_dir + 'valid_list.txt', sorted(valid_list))
    save_string_list(save_dir + 'test_list.txt', sorted(test_list))
    save_string_list(save_dir + 'valid_test_list.txt', sorted(valid_list + test_list))


if __name__ == '__main__':
    # proc_main()
    split_list()

