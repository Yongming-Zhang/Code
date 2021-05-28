# -*- coding: utf-8 -*-
import cv2
import struct
from cta_util import *
from scipy import ndimage
from lib.utils.contour import find_contours, fill_contours, draw_contours


def load_cpr_coord_map(file_path):
    f = open(file_path, mode='rb')
    a = f.read()
    w, h, c, t = struct.unpack('iiii', a[:16])
    assert c == 3 and t == 10, 'The third and fourth items of cpr coor map should be 3 and 10'
    maps = struct.unpack('f' * w * h * c, a[16:])
    maps = numpy.float32(maps).reshape(h, w, c)
    return maps


class CtaCprMapper(object):
    def __init__(self, cpr_coord_maps):
        self.cpr_coord_maps = cpr_coord_maps

    def cta2cpr(self, cta_data, mode='nearest'):
        # assert mode == 'nearest', 'Only support nearest interplotation'
        cta_d, cta_h, cta_w = cta_data.shape[:3]
        cpr_data_list = []
        for cpr_coord_map in self.cpr_coord_maps:
            cpr_h, cpr_w = cpr_coord_map.shape[:2]
            if mode == 'nearest':
                cpr_data = numpy.ones([cpr_h, cpr_w] + list(cta_data.shape[3:]), cta_data.dtype) * -1024
                for i in range(cpr_h):
                    for j in range(cpr_w):
                        x, y, z = cpr_coord_map[i, j]
                        x, y, z = int(round(x)), int(round(y)), int(round(z))
                        if 0 <= x < cta_w and 0 <= y < cta_h and 0 <= z < cta_d:
                            cpr_data[i, j] = cta_data[z, y, x]
            else:
                xs, ys, zs = cpr_coord_map[:, :, 0], cpr_coord_map[:, :, 1], cpr_coord_map[:, :, 2]
                cpr_data = ndimage.map_coordinates(cta_data, [zs, ys, xs], order=1, cval=-1024).astype(cta_data.dtype)
            cpr_data_list.append(cpr_data)
        return cpr_data_list


def map_cta_ann_json_to_cpr(cpr_coord_maps, cta_ann_json_dict, cta_shape, cta_instance_dict, cpr_instances):
    mapper = CtaCprMapper(cpr_coord_maps)
    cpr_json_dict = {
        'patientID': cta_ann_json_dict['patientID'],
        'studyUID': cta_ann_json_dict['studyUID'],
        'other_info': {'doctorId': cta_ann_json_dict['other_info']['doctorId']},
        'nodes': []
    }
    for node in cta_ann_json_dict['nodes']:
        if isinstance(node['descText'][1][0][0]['select'], list):
            continue
        les_type = g_ccta_lesion_type_dict[node['descText'][1][0][0]['select']]
        stenosis = g_stenosis_type_dict[node['descText'][1][1][0]['select']]
        detect_necessity = g_detect_necessity_dict[node['descText'][1][2][0]['select']]
        segments = ['other' if s == u'其他' else s for s in node['descText'][1][3][0]['select']]
        if les_type == 'none':
            continue
        new_node = {'type': les_type, 'detect_necessity': detect_necessity, 'segments': segments}
        if les_type in ['stent_clear', 'stent_unclear', 'stent_block', 'stent_none', 'mca', 'mca_mb']:
            pts3d = []
            for roi in node['rois']:
                z = cta_instance_dict[roi['slice_index']]
                x, y = roi['edge'][0]
                pts3d.append([x, y, z])
            new_node['pts3d'] = pts3d
        else:
            new_node['stenosis'] = stenosis
            cta_mask = numpy.zeros(cta_shape, dtype='uint8')
            for roi in node['rois'] + node['bounds']:
                contours_numpy = numpy.int32(roi['edge'])
                z = cta_instance_dict[roi['slice_index']]
                cv2.drawContours(cta_mask[z], (contours_numpy,), 0, color=255, thickness=-1)
            cpr_masks = mapper.cta2cpr(cta_mask, 'trilinear')
            rois = []
            for i, cpr_mask in enumerate(cpr_masks):
                edge_list = find_contours(cpr_mask)
                for edge in edge_list:
                    rois.append({'slice_index': cpr_instances[i], 'edge': edge[:, 0, :].tolist()})
            if len(rois) == 0:
                new_node = None
            else:
                new_node['rois'] = rois
        if new_node is not None:
            cpr_json_dict['nodes'].append(new_node)
    return cpr_json_dict