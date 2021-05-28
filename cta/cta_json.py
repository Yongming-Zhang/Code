# -*- coding: utf-8 -*-
import cv2
import numpy
import random
from cta.cta_util import g_stenosis_type_dict, g_detect_necessity_dict, g_ccta_lesion_type_dict


def gen_cta_psid_json_dict(cta_json_dict_list, using_series_id=False):
    cta_psid_json_dicts = {}
    for cta_json_dict in cta_json_dict_list:
        if cta_json_dict['other_info']['passReason'] != '':
            continue
        if len(cta_json_dict['nodes']) == 0:
            continue
        psid = cta_json_dict['patientID'] + '_' + cta_json_dict['studyUID']
        if using_series_id:
            psid += ('_' + cta_json_dict['seriesUID'])
        if psid not in cta_psid_json_dicts:
            cta_psid_json_dicts[psid] = cta_json_dict
        else:
            cta_psid_json_dicts[psid]['nodes'] += cta_json_dict['nodes']
    return cta_psid_json_dicts


def is_ann_by_endpoint(les_type):
    return les_type in ['stent_clear', 'stent_unclear', 'stent_block', 'stent_none', 'mca', 'mca_mb']


def is_ann_by_point(roi):
    edge = roi['edge']
    if len(edge) == 2:
        x, y = edge[-1]
        return x == 1 and y == 1
    return False


def get_cta_nodes(cta_json_dict, cta_instance_dict, max_pts_num=1000):
    nodes = []
    for node in cta_json_dict['nodes']:
        if 'descText' in node:
            if isinstance(node['descText'][1][0][0]['select'], list):
                continue
            les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
            stenosis = g_stenosis_type_dict[node['descText'][1][1][0]['select']]
            if isinstance(node['descText'][1][2][0]['select'], list):
                detect_necessity = 'none'
                segments = node['descText'][1][2][0]['select']
            else:
                detect_necessity = g_detect_necessity_dict[node['descText'][1][2][0]['select']]
                segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
            if les_type == 'none':
                continue
        else:
            les_type = node['merged_type']
            detect_necessity = 'none'
            segments = []
            stenosis = 'none'
        new_node = {'type': les_type, 'detect_necessity': detect_necessity, 'segments': segments, 'stenosis': stenosis}
        pts3d = []
        # if is_ann_by_endpoint(les_type):
        #     for roi in node['rois']:
        #         z = cta_instance_dict[roi['slice_index']]
        #         x, y = roi['edge'][0]
        #         pts3d.append([x, y, z])
        is_ann_by_endpoint_flag = False
        if 'bounds' not in node:
            node['bounds'] = []
        for roi in node['rois'] + node['bounds']:
            z = cta_instance_dict[roi['slice_index']]
            if is_ann_by_point(roi):
                x, y = roi['edge'][0]
                pts3d.append([x, y, z])
                is_ann_by_endpoint_flag = True
            elif len(roi['edge']) == 1:
                x, y = roi['edge'][0]
                pts3d.append([x, y, z])
            else:
                contours_numpy = numpy.int32(roi['edge'])
                cta_mask = numpy.zeros([512, 512], dtype='uint8')
                cv2.drawContours(cta_mask, (contours_numpy,), 0, color=255, thickness=-1)
                ys, xs = cta_mask.nonzero()
                for x, y in zip(xs, ys):
                    pts3d.append([x, y, z])
            if len(pts3d) > max_pts_num > 0:
                pts3d = random.sample(pts3d, max_pts_num)
        new_node['is_ann_by_endpoint'] = is_ann_by_endpoint_flag
        if len(pts3d) > 0:
            new_node['pts3d'] = pts3d
            nodes.append(new_node)
    return nodes

