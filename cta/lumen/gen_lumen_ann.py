# -*- coding: utf-8 -*-
from lumen_util import *
import cv2
import numpy
import random
from collections import namedtuple
from lib.utils.mio import *
from lib.utils.util import pairwise_dist2
from cta.cta_util import g_stenosis_type_dict, g_detect_necessity_dict, g_ccta_lesion_type_dict
import glob


Vessel = namedtuple('Vessel', 'name c3d lumen')
C3dMapper = namedtuple('C3dMapper', 'pts3d vessel dist2_mat c3d_nearest_indices c3d_nearest_dist2s')


def gen_cta_psid_json_dict(cta_json_dict_list):
    cta_psid_json_dicts = {}
    for cta_json_dict in cta_json_dict_list:
        if cta_json_dict['other_info']['passReason'] != '':
            continue
        if len(cta_json_dict['nodes']) == 0:
            continue
        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        if psid not in cta_psid_json_dicts:
            cta_psid_json_dicts[psid] = cta_json_dict
        else:
            cta_psid_json_dicts[psid]['nodes'] += cta_json_dict['nodes']
    return cta_psid_json_dicts


def is_ann_by_endpoint(les_type):
    return les_type in ['stent_clear', 'stent_unclear', 'stent_block', 'stent_none', 'mca', 'mca_mb']


def get_cta_nodes(cta_json_dict, cta_instance_dict):
    max_pts_num = 1000
    nodes = []
    for node in cta_json_dict['nodes']:
        if isinstance(node['descText'][1][0][0]['select'], list):
            continue
        les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
        stenosis = g_stenosis_type_dict[node['descText'][1][1][0]['select']]
        detect_necessity = g_detect_necessity_dict[node['descText'][1][2][0]['select']]
        segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
        if les_type == 'none':
            continue
        new_node = {'type': les_type, 'detect_necessity': detect_necessity, 'segments': segments}
        pts3d = []
        if is_ann_by_endpoint(les_type):
            for roi in node['rois']:
                z = cta_instance_dict[roi['slice_index']]
                x, y = roi['edge'][0]
                pts3d.append([x, y, z])
        else:
            new_node['stenosis'] = stenosis
            for roi in node['rois'] + node['bounds']:
                contours_numpy = numpy.int32(roi['edge'])
                z = cta_instance_dict[roi['slice_index']]
                cta_mask = numpy.zeros([512, 512], dtype='uint8')
                cv2.drawContours(cta_mask, (contours_numpy,), 0, color=255, thickness=-1)
                ys, xs = cta_mask.nonzero()
                for x, y in zip(xs, ys):
                    pts3d.append([x, y, z])
            if len(pts3d) > max_pts_num:
                pts3d = random.sample(pts3d, max_pts_num)
        if len(pts3d) > 0:
            new_node['pts3d'] = pts3d
            nodes.append(new_node)
    return nodes


def get_c3d_mapper(pts3d, vessel):
    d2_mat = pairwise_dist2(pts3d, vessel.c3d)
    c3d_nearest_indices = d2_mat.argmin(axis=1)
    c3d_nearest_dist2s = d2_mat.min(axis=1)
    mapper = C3dMapper(
        pts3d=pts3d, vessel=vessel,
        dist2_mat=d2_mat, c3d_nearest_indices=c3d_nearest_indices, c3d_nearest_dist2s=c3d_nearest_dist2s)
    return mapper


def find_best_vessel(c3d_mappers):
    min_dist, min_dist_mapper = 300, None
    for c3d_mapper in c3d_mappers:
        d = c3d_mapper.c3d_nearest_dist2s.max()
        # print('    ', c3d_mapper.vessel.name, c3d_mapper.c3d_nearest_dist2s.shape[0], d, c3d_mapper.c3d_nearest_dist2s.mean())
        if d < min_dist:
            min_dist = d
            min_dist_mapper = c3d_mapper
    return min_dist_mapper


def map_cta_node_to_lumen_indices(cta_node, vessels):
    max_valid_dist2, max_c3d_dist2 = 50, 5
    node_pts3d = numpy.float32(cta_node['pts3d'])
    c3d_mappers = [get_c3d_mapper(node_pts3d, vessel) for vessel in vessels]
    # print('  Node %d, %s' % (cta_node['node_index'], str(cta_node['segments'])))
    min_dist_mapper = find_best_vessel(c3d_mappers)
    if min_dist_mapper is None:
        return []
    i0 = min_dist_mapper.c3d_nearest_indices.min()
    i1 = min_dist_mapper.c3d_nearest_indices.max()
    if is_ann_by_endpoint(cta_node['type']):
        node_pts3d = numpy.float32(min_dist_mapper.vessel.c3d[i0:(i1+1)])
    else:
        mapped_c3d_pts = []
        node_pts_nearest_dist2s = min_dist_mapper.dist2_mat.min(axis=0)
        for i in range(i0, i1 + 1):
            if node_pts_nearest_dist2s[i] < max_valid_dist2:
                mapped_c3d_pts.append(min_dist_mapper.vessel.c3d[i])
        if len(mapped_c3d_pts) == 0:
            return []
        node_pts3d = numpy.float32(mapped_c3d_pts)
    c3d_mappers = [get_c3d_mapper(node_pts3d, vessel) for vessel in vessels]
    lumen_indices = []
    for c3d_mapper in c3d_mappers:
        i0 = c3d_mapper.c3d_nearest_indices.min()
        i1 = c3d_mapper.c3d_nearest_indices.max()
        node_pts_nearest_dist2s = c3d_mapper.dist2_mat.min(axis=0)
        node_pts_nearest_indices = c3d_mapper.dist2_mat.argmin(axis=0)
        for i in range(i0, i1 + 1):
            if node_pts_nearest_dist2s[i] < max_c3d_dist2:
                li = c3d_mapper.vessel.lumen[i]
                lumen_indices.append(li)
                c3d_str = ','.join(['%.2f' % x for x in c3d_mapper.vessel.c3d[i]])
                p3d_str = ','.join(['%.2f' % x for x in node_pts3d[node_pts_nearest_indices[i]]])
                if node_pts_nearest_dist2s[i] > 20:
                    print('  Node %d, %s, c3d=%d(%s), lumen=%d, min_dist2=%.2f, min_node_pt=(%s)' % (
                        cta_node['node_index'], c3d_mapper.vessel.name, i, c3d_str, li, node_pts_nearest_dist2s[i], p3d_str))
    return sorted(list(set(lumen_indices)))


def draw_node(node, lumen_vessel_dict, lumen_img_dict, save_dir):
    num_pad_imgs, num_pad_pixels = 10, 10
    save_name_prefix = 'n%d_' % node['node_index'] if 'node_index' in node else ''
    save_name_prefix += node['type']
    for k in ['detect_necessity', 'stenosis']:
        if k in node:
            save_name_prefix += '_' + node[k]
    if 'segments' in node:
        save_name_prefix += ('_' + '~'.join(node['segments']))
    lumen_h, lumen_w = lumen_img_dict.values()[0].shape
    empty_img = numpy.zeros([lumen_h, lumen_w], dtype='uint8')
    cv2.line(empty_img, (0, 0), (lumen_w - 1, lumen_h - 1), 255, 2)
    cv2.line(empty_img, (lumen_w - 1, 0), (0, lumen_h - 1), 255, 2)
    pad_img = numpy.zeros([lumen_h, num_pad_pixels], dtype='uint8')
    mkdir_safe(save_dir)
    for lumen_index in node['lumen_indices']:
        imgs = []
        for i in range(lumen_index - num_pad_imgs, lumen_index + num_pad_imgs + 1):
            imgs.append(lumen_img_dict.get(i, empty_img))
            imgs.append(pad_img)
        draw_img = numpy.hstack(imgs[:-1])
        i0 = (lumen_w + num_pad_pixels) * num_pad_imgs - 3
        draw_img[:, i0:(i0+3)] = 255
        i0 = (lumen_w + num_pad_pixels) * num_pad_imgs + lumen_w
        draw_img[:, i0:(i0+3)] = 255
        save_path = save_dir + save_name_prefix + '_%s_l%d.png' % (lumen_vessel_dict[lumen_index], lumen_index)
        save_image(save_path, draw_img)
        # print('    %s %dx%d' % (save_path, draw_img.shape[0], draw_img.shape[1]))


def gen_lumen_json_main():
    base_dir = '/data2/zhangfd/data/cta/b2_407/cta2cpr/'
    lumen_dir = base_dir + 'cpr_lumen_s9_n20_reorg/'
    cta_dcm_dir = base_dir + 'cta_dicom/'
    cta_ann_json_dir = base_dir + 'annotation/'
    center_crop, rescale = 32, 128
    lumen_json_dir = base_dir + 'annotation/lumen/json/'
    lumen_patch_dir = base_dir + 'annotation/lumen/patches_crop%d_scale%d/' % (center_crop, rescale)
    mkdir_safe(lumen_json_dir)
    mkdir_safe(lumen_patch_dir)
    cta_json_dict_list = json_load(cta_ann_json_dir + 'task_1265_0-1588272710.json')
    cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    is_draw_patch = True
    # psid_list = ['0075808_1.2.392.200036.9116.2.5.1.37.2417525831.1568698731.788682']
    # psid_list = ['0366427_1.2.392.200036.9116.2.5.1.37.2417525831.1551675244.770877']
    psid_list = ['0784311_1.2.392.200036.9116.2.5.1.37.2417525831.1548804688.464916']
    is_skip = True
    for psid in sorted(cta_psid_json_dicts.keys()):
        cta_json_dict = cta_psid_json_dicts[psid]
        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        if psid not in psid_list and is_skip:
            continue
        # is_skip = False
        cpr_series_dirs = sorted(glob.glob(lumen_dir + psid + '/CPR_*/'))
        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + psid + '/*.dcm'))
        cta_instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(cta_dcm_paths)}
        cta_ann_nodes = get_cta_nodes(cta_json_dict, cta_instance_dict)
        vessels, new_nodes = [], []
        for cpr_series_dir in cpr_series_dirs:
            vessel_name = cpr_series_dir.split('/')[-2].split('_')[-1]
            vessel_jd = json_load(cpr_series_dir + vessel_name + '_000.dcm.json')
            vessel = Vessel(name=vessel_name, c3d=numpy.float32(vessel_jd['centerline3d']), lumen=vessel_jd['spin_slice'])
            vessels.append(vessel)
        print('%s: %d nodes' % (psid, len(cta_ann_nodes)))
        for node_index, cta_node in enumerate(cta_ann_nodes):
            cta_node['node_index'] = node_index
            lumen_indices = map_cta_node_to_lumen_indices(cta_node, vessels)
            cta_node['lumen_indices'] = lumen_indices
            new_nodes.append(cta_node)
        new_json = cta_json_dict.copy()
        new_json['nodes'] = new_nodes
        json_save(lumen_json_dir + '%s.json' % psid, new_json)
        if is_draw_patch:
            lumen_img_dict = load_lumen_as_dict(lumen_dir + psid + '/LUMEN/', center_crop, rescale)
            lumen_vessel_dict = {}
            for vessel in vessels:
                for li in vessel.lumen:
                    lumen_vessel_dict[li] = vessel.name
            for node in new_json['nodes']:
                print('  %s %s %d lumen' % (node['type'], str(node['segments']), len(node['lumen_indices'])))
                draw_node(node, lumen_vessel_dict, lumen_img_dict, lumen_patch_dir + psid + '/')


if __name__ == '__main__':
    gen_lumen_json_main()

