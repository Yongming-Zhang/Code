# -*- coding: utf-8 -*-
from lumen_util import *
import os
import cv2
import numpy
import random
from collections import namedtuple, defaultdict, Counter
from lib.utils.mio import *
from lib.utils.util import pairwise_dist2
from cta.cta_util import load_vessel_name_map, get_cta_pixel_spacings
from cta.cta_json import gen_cta_psid_json_dict, is_ann_by_endpoint, get_cta_nodes
from lib.image.draw import draw_texts
from lib.image.image import image_center_crop
import glob


Vessel = namedtuple('Vessel', 'name c3d lumen lids')
C3dMapper = namedtuple('C3dMapper', 'pts3d vessel dist2_mat c3d_nearest_indices c3d_nearest_dist2s')
center_crop, rescale = 40, 80


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
        if d < min_dist:
            min_dist = d
            min_dist_mapper = c3d_mapper
    return min_dist_mapper


def map_cta_node_to_lumen_indices(cta_node, vessels):
    max_valid_dist2, max_c3d_dist2 = 50, 5
    node_pts3d = numpy.float32(cta_node['pts3d'])
    # print(cta_node['pts3d'])
    c3d_mappers = [get_c3d_mapper(node_pts3d, vessel) for vessel in vessels]
    # print('  Node %d, %s' % (cta_node['node_index'], str(cta_node['segments'])))
    min_dist_mapper = find_best_vessel(c3d_mappers)
    if min_dist_mapper is None:
        return {}
    i0 = min_dist_mapper.c3d_nearest_indices.min()
    i1 = min_dist_mapper.c3d_nearest_indices.max()
    # if is_ann_by_endpoint(cta_node['type']):
    if cta_node['is_ann_by_endpoint']:
        node_pts3d = numpy.float32(min_dist_mapper.vessel.c3d[i0:(i1+1)])
    else:
        mapped_c3d_pts = []
        node_pts_nearest_dist2s = min_dist_mapper.dist2_mat.min(axis=0)
        for i in range(i0, i1 + 1):
            if node_pts_nearest_dist2s[i] < max_valid_dist2:
                mapped_c3d_pts.append(min_dist_mapper.vessel.c3d[i])
        if len(mapped_c3d_pts) == 0:
            return {}
        node_pts3d = numpy.float32(mapped_c3d_pts)
    c3d_mappers = [get_c3d_mapper(node_pts3d, vessel) for vessel in vessels]
    lumen_indices_dict = {}
    for c3d_mapper in c3d_mappers:
        lumen_indices = []
        i0 = c3d_mapper.c3d_nearest_indices.min()
        i1 = c3d_mapper.c3d_nearest_indices.max()
        node_pts_nearest_dist2s = c3d_mapper.dist2_mat.min(axis=0)
        node_pts_nearest_indices = c3d_mapper.dist2_mat.argmin(axis=0)
        for i in range(i0, i1 + 1):
            if node_pts_nearest_dist2s[i] < max_c3d_dist2:
                li = c3d_mapper.vessel.lids[i]
                lumen_indices.append(li)
                c3d_str = ','.join(['%.2f' % x for x in c3d_mapper.vessel.c3d[i]])
                p3d_str = ','.join(['%.2f' % x for x in node_pts3d[node_pts_nearest_indices[i]]])
                if node_pts_nearest_dist2s[i] > 20:
                    print('  Node %d, %s, c3d=%d(%s), lumen=%d, min_dist2=%.2f, min_node_pt=(%s)' % (
                        cta_node['node_index'], c3d_mapper.vessel.name, i, c3d_str, li, node_pts_nearest_dist2s[i], p3d_str))
        lumen_indices = sorted(list(set(lumen_indices)))
        lumen_indices_dict[c3d_mapper.vessel.name] = lumen_indices
    return lumen_indices_dict


def load_vessel(cpr_lumen_dir, spacing_x, spacing_y, spacing_z, vessel_id, vessel_name_map):
    centerline3d = json_load(cpr_lumen_dir + 'cpr/centerline/%s.3d' % vessel_id)
    c3d = centerline3d['points']
    lumen_img_dict = load_lumen_as_dict(cpr_lumen_dir + 'lumen/%s/' % vessel_id, center_crop, rescale)
    min_lumen_id = min(lumen_img_dict.keys())
    lumen_indices = get_centerline3d_lumen_indices(c3d, spacing_x, spacing_y, spacing_z, start_i=min_lumen_id)
    vessel = Vessel(vessel_name_map.get(vessel_id, vessel_id), numpy.float32(c3d), lumen_img_dict, lumen_indices)
    return vessel


def load_all_vessels(cpr_lumen_dir, spacing_x, spacing_y, spacing_z, vessel_name_map, valid_vessels=None):
    c3d_paths = sorted(glob.glob(cpr_lumen_dir + 'cpr/centerline/*.3d'))
    vessels = []
    for c3d_path in c3d_paths:
        vessel_id = c3d_path.split('/')[-1][:-3]
        if valid_vessels is not None and vessel_id not in valid_vessels:
            continue
        vessel = load_vessel(cpr_lumen_dir, spacing_x, spacing_y, spacing_z, vessel_id, vessel_name_map)
        vessels.append(vessel)
    return vessels


def draw_one_vessel_patch(vessel, lumen_types):
    draw_imgs = []
    img_head_h = 20
    lumen_h, lumen_w = list(vessel.lumen.values())[0].shape
    draw_lumen_h = img_head_h + lumen_h
    for li in sorted(vessel.lumen.keys()):
        labels = sorted(list(set([g_label_dict[lumen_type] for lumen_type in lumen_types[li]])))
        label_str = ''.join([str(label) for label in labels])
        lumen_img = vessel.lumen[li]
        draw_img = numpy.vstack([numpy.zeros([img_head_h, lumen_w], dtype='uint8'), lumen_img])
        draw_texts(draw_img, (0, img_head_h + 1), [label_str], 255, is_bold=False, thick=1, font_scale=0.4)
        draw_imgs.append(draw_img)
    num_per_row = 30
    num_rows = int(math.ceil(len(draw_imgs) * 1. / num_per_row))
    canvas = numpy.zeros([draw_lumen_h * num_rows, lumen_w * num_per_row], dtype='uint8')
    for di, draw_img in enumerate(draw_imgs):
        i, j = di // num_per_row, di % num_per_row
        x, y = j * lumen_w, i * draw_lumen_h
        canvas[y:(y+draw_lumen_h), x:(x+lumen_w)] = draw_img
    return canvas


def draw_patches(new_nodes, vessels, save_dir):
    mkdir_safe(save_dir)
    lumen_types = defaultdict(lambda: defaultdict(list))
    for node in new_nodes:
        node_type = node['type'].split('_')[0]
        if node_type == 'mca':
            continue
        for vessel_name, lids in node['lumen_indices'].items():
            for li in lids:
                lumen_types[vessel_name][li].append(node_type)
    for vessel in vessels:
        draw_img = draw_one_vessel_patch(vessel, lumen_types[vessel.name])
        save_image(save_dir + vessel.name + '.png', draw_img)


def load_dataset(name):
    if name == 'b1':
        base_dir = '/breast_data/cta/new_data/b1_bt_587/'
        ann_json_paths = sorted(glob.glob(base_dir + 'annotations/ann_pts3d_for_check/*/*/*/ann.json'))
        cta_psid_json_dicts = {}
        for ann_json_path in ann_json_paths:
            pid, study, series = ann_json_path.split('/')[-4:-1]
            cta_psid_json_dicts[pid + '_' + study] = json_load(ann_json_path)
    elif name == 'b2':
        base_dir = '/breast_data/cta/new_data/b2_sg_407/'
        cta_json_dict_list = json_load(base_dir + 'annotation/task_1265_0-1588272710.json')
        cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    else:
        base_dir = '/breast_data/cta/new_data/b3_azcto_150/'
        cta_json_dict_list = json_load(base_dir + 'annotation/task_1830_0-1599383101.json')
        cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list, True)
    return base_dir, cta_psid_json_dicts


def gen_lumen_json_main():
    # base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    dataset_name = 'b1'
    base_dir, cta_psid_json_dicts = load_dataset(dataset_name)
    cpr_lumen_dir = base_dir + 'raw/cpr_lumen_s9_n20/'
    cta_dcm_dir = base_dir + 'cta_dicom/'
    lumen_json_dir = base_dir + 'annotation/lumen_v2/json/'
    lumen_patch_dir = base_dir + 'annotation/lumen_v2/patches_crop%d_scale%d/' % (center_crop, rescale)
    mkdir_safe(lumen_json_dir)
    mkdir_safe(lumen_patch_dir)
    is_draw_patch = True
    # psid_list = ['0784311_1.2.392.200036.9116.2.5.1.37.2417525831.1548804688.464916']
    # psid_list = ['0170533_1.2.392.200036.9116.2.5.1.37.2417525831.1548375430.185550']
    is_skip = True
    vessel_name_map = load_vessel_name_map()
    for psid in sorted(cta_psid_json_dicts.keys()):
        cta_json_dict = cta_psid_json_dicts[psid]
        # psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        # if psid not in psid_list and is_skip:
        #     continue
        is_skip = False
        spacing_x, spacing_y, spacing_z = get_cta_pixel_spacings(cta_dcm_dir + psid + '/')
        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + psid + '/*.dcm'))
        cta_instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(cta_dcm_paths)}
        cta_ann_nodes = get_cta_nodes(cta_json_dict, cta_instance_dict)
        valid_vessels = None
        if dataset_name == 'b3':
            gen_json_paths = glob.glob(base_dir + 'annotation/cpr_s9_n20_reorg/raw_json/%s_*.json' % psid)
            valid_vessels = set([gen_json_path.split('_')[-1][:-5] for gen_json_path in gen_json_paths])
        vessels = load_all_vessels(
            cpr_lumen_dir + psid + '/', spacing_x, spacing_y, spacing_z, vessel_name_map, valid_vessels)
        print('%s: %d nodes, %d vessels' % (psid, len(cta_ann_nodes), len(vessels)))
        if len(vessels) == 0:
            continue
        new_nodes = []
        for node_index, cta_node in enumerate(cta_ann_nodes):
            cta_node['node_index'] = node_index
            lumen_indices_dict = map_cta_node_to_lumen_indices(cta_node, vessels)
            cta_node['lumen_indices'] = lumen_indices_dict
            new_nodes.append(cta_node)
            num_lumen_indices = sum([len(x) for x in lumen_indices_dict.values()])
            print('  %s %s %d lumen' % (cta_node['type'], str(cta_node['segments']), num_lumen_indices))
        new_json = cta_json_dict.copy()
        new_json['nodes'] = new_nodes
        # print(new_json)
        json_save(lumen_json_dir + '%s.json' % psid, new_json)
        if is_draw_patch:
            draw_patches(new_nodes, vessels, lumen_patch_dir + psid + '/')


def load_and_crop_lumen_images(lumen_dir, lumen_windows, patch_size_xy):
    lumen_paths = sorted(glob.glob(lumen_dir + '*.dcm'))
    d = {}
    for lumen_path in lumen_paths:
        instance_num = int(lumen_path.split('/')[-1][:-4])
        imgs = gen_image_from_dcm(lumen_path, lumen_windows)
        rgb_img = numpy.zeros([patch_size_xy, patch_size_xy, 3], dtype='uint8')
        for c in range(3):
            rgb_img[:, :, c] = image_center_crop(imgs[c], patch_size_xy, patch_size_xy)
        d[instance_num] = rgb_img
    return d


def load_psid_set_dict(base_dir):
    psid_set_dict = {}
    for set_name in ['train', 'valid', 'test']:
        psid_list = load_string_list(base_dir + 'config/%s_psid_list.txt' % set_name)
        for psid in psid_list:
            psid_set_dict[psid] = set_name
    return psid_set_dict


def get_label(lumen_types):
    labels = list(set([g_label_dict[lumen_type] for lumen_type in lumen_types]))
    if len(labels) == 0:
        return 0
    if len(labels) == 1:
        return labels[0]
    if 4 in labels:
        return 4
    return 3


def gen_proccessed_data():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    cpr_lumen_dir = base_dir + 'raw/cpr_lumen_s9_n20/'
    lumen_json_dir = base_dir + 'annotation/lumen_v2/json/'
    save_base_dir = base_dir + 'model_data/lumen_v2/layer_seg/'
    list_save_dir = save_base_dir + 'lists/'
    mkdir_safe(list_save_dir)
    patch_size_xy = 60
    lumen_img_base_save_dir = '/data3/ccta/plaque/b2_sg_407/model_data/lumen_v2/layer_seg/'
    lumen_img_save_dir = lumen_img_base_save_dir + 'image_win1_p%d/' % patch_size_xy
    lumen_msk_save_dir = lumen_img_base_save_dir + 'layer_mask/'
    mkdir_safe(lumen_img_save_dir)
    mkdir_safe(lumen_msk_save_dir)
    lumen_windows = [(300, 700), (-100, 700), (-500, 1000)]
    set_img_lists = defaultdict(list)
    vessel_name_map = load_vessel_name_map()
    psid_set_dict = load_psid_set_dict(base_dir)
    label_name_dict = {v: k for (k, v) in g_label_dict.items()}
    set_names = ['train', 'valid', 'test']
    set_label_nums = {set_name: [0, 0, 0, 0, 0] for set_name in ['train', 'valid', 'test']}
    for psid in sorted(psid_set_dict.keys()):
        if not os.path.exists(lumen_json_dir + '%s.json' % psid):
            continue
        json_dict = json_load(lumen_json_dir + '%s.json' % psid)
        lumen_types = defaultdict(lambda: defaultdict(list))
        for node in json_dict['nodes']:
            node_type = node['type'].split('_')[0]
            if node_type == 'mca':
                continue
            for vessel_name, lids in node['lumen_indices'].items():
                for li in lids:
                    lumen_types[vessel_name][li].append(node_type)
        set_name = psid_set_dict[psid]
        for lumen_dir in sorted(glob.glob(cpr_lumen_dir + psid + '/lumen/*/')):
            vessel_id = lumen_dir.split('/')[-2]
            vessel_name = vessel_name_map.get(vessel_id, vessel_id)
            lumen_img_dict = load_and_crop_lumen_images(lumen_dir, lumen_windows, patch_size_xy)
            lumen_ids = sorted(lumen_img_dict.keys())
            img_tensor = numpy.hstack([lumen_img_dict[lumen_id] for lumen_id in lumen_ids])
            layer_labels = [get_label(lumen_types[vessel_name][lumen_id]) for lumen_id in lumen_ids]
            for label in layer_labels:
                set_label_nums[set_name][label] += 1
            mask_tensor = numpy.uint8(layer_labels)
            img_name = psid + '_' + vessel_name
            set_img_lists[set_name].append(img_name)
            save_image(lumen_img_save_dir + img_name + '.png', img_tensor)
            save_image(lumen_msk_save_dir + img_name + '.png', mask_tensor)
            label_counter = Counter(layer_labels)
            label_str = ', '.join(['%s=%d' % (label_name_dict[label], num) for label, num in label_counter.most_common()])
            print('%s: %s' % (img_name, label_str))
    for set_name in set_names:
        save_string_list(list_save_dir + '%s_img_list.txt' % set_name, set_img_lists[set_name])
        label_str = ', '.join(['%s=%d' % (label_name_dict[label], set_label_nums[set_name][label]) for label in range(5)])
        print('%s: %d imgs, label: %s' % (set_name, len(set_img_lists[set_name]), label_str))


if __name__ == '__main__':
    gen_lumen_json_main()
    # gen_proccessed_data()

