# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../')
import os
import cv2
import glob
import math
import numpy
import random
from lib.utils.union_find import UnionFind
from lib.utils.rect import rect_iou, rect_union
from lib.utils.util import pad_string, pairwise_dist2
from lib.utils.mio import json_load, dicom_load
import SimpleITK as sitk


g_doctor_id_names = {
    # cta
    41: 'lihaibo_sg',
    64: 'panjun_sg',
    85: 'hatingting_sg',
    122: 'lixinhua_sg',
    156: 'zhangyuanfang_kf',
    174: 'tanglili_sg',
    175: 'liyingming_sg',
    220: 'weiyanlei_dx',
    224: 'caobaoqing_my',
    235: 'panjun_sg',
    263: 'tanglili'
}


g_stenosis_type_dict = {
    u'无明显狭窄': 'none',
    u'轻微（1%-24%）': 'lower',
    u'轻度（25%-49%）': 'low',
    u'中度（50%-69%）': 'mid',
    u'重度（70%-99%）': 'high',
    u'闭塞（100%）': 'total',
    '': 'none'
}


g_ccta_lesion_type_dict = {
    u'钙化': 'cal',
    u'混合': 'mix',
    u'低密度': 'low',
    u'闭塞': 'block',
    u'支架通畅': 'stent_clear',
    u'支架欠通畅': 'stent_unclear',
    u'支架闭塞': 'stent_block',
    u'支架评价受限': 'stent_none',
    u'支架': 'stent_none',
    u'壁冠状动脉': 'mca',  # mural coronary artery
    u'壁冠状动脉-心肌桥': 'mca_mb',  # MCA-MB
    u'心肌桥': 'mca_mb',  # MCA-MB
    u'无问题': 'none',
    '': 'cal'
}


g_detect_necessity_dict = {
    u'不用检出': 'not',
    u'可检可不检': 'ok',
    u'最好检出': 'better',
    u'必须检出': 'must',
    '': 'none'
}

g_stenosis_type_list = ['none', 'lower', 'low', 'mid', 'high', 'total']


def load_vessel_name_map():
    map_path = '/breast_data/cta/new_data/map_coro_idx.txt'
    map_path08 = '/data2/zhangfd/code/cta/map_coro_idx.txt'
    return json_load(map_path if os.path.exists(map_path) else map_path08)


def get_bbox(pts):
    pts = numpy.float32(pts)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    return [x0, y0, w, h]


def get_size(pts):
    pc, (w, h), theta = cv2.minAreaRect(numpy.float32(pts))
    return w, h


def fmt_ratio_str(numerator, denominator):
    return '%.2f%%(%d/%d)' % (numerator * 100. / (denominator + 1e-5), numerator, denominator)


def get_pred_bbox_list(pred_list, score_thresh, is_combine=False):
    pred_bbox_list = [pred for pred in pred_list if pred['score'] >= score_thresh]
    if not is_combine:
        return pred_bbox_list
    id_pairs = []
    for i, bboxi in enumerate(pred_bbox_list):
        for j, bboxj in enumerate(pred_bbox_list):
            if rect_iou(bboxi['bbox'], bboxj['bbox'], 'iomin') >= 0.5:
                id_pairs.append((i, j))
    uf = UnionFind(id_pairs)
    group_ids_list = uf.run()
    merged_bbox_list = []
    for ids in group_ids_list:
        labels = list(set([pred_bbox_list[i]['category_id'] for i in ids]))
        pred_dict = {
            'bbox': rect_union([pred_bbox_list[i]['bbox'] for i in ids]),
            'score': max([pred_bbox_list[i]['score'] for i in ids]),
            'category_id': labels[0] if len(labels) == 1 else 3
        }
        merged_bbox_list.append(pred_dict)
    return merged_bbox_list


def gen_table_strings(value_dict, table_name='', col_name_len=None, col_unit_len=None, align='c', row_names=None, col_names=None):
    if row_names is None:
        row_names = sorted(value_dict.keys())
    if col_names is None:
        col_names = []
        for k1, vd in value_dict.items():
            col_names += vd.keys()
        col_names = sorted(list(set(col_names)))
    if col_name_len is None:
        col_name_len = max([len(row_name.encode('gbk')) for row_name in row_names])
        col_name_len = max(col_name_len, len(table_name))
    if col_unit_len is None:
        col_unit_len = []
        for col_name in col_names:
            col_lens = [len(col_name)] + [len(str(value_dict[row_name][col_name])) for row_name in row_names]
            col_unit_len.append(max(col_lens) + 2)
    elif type(col_unit_len) not in [list, tuple]:
        col_unit_len = [col_unit_len] * len(col_names)
    lines = ['|'.join(['', pad_string(table_name, col_name_len, align=align)] +
                      [pad_string(cn, cl, align=align) for (cn, cl) in zip(col_names, col_unit_len)]) + '|']
    for row_name in row_names:
        line_items = [pad_string(row_name, col_name_len, align=align)]
        line_items += [pad_string(str(value_dict[row_name][cn]), cl, align=align) for (cn, cl) in zip(col_names, col_unit_len)]
        lines.append('|' + '|'.join(line_items) + '|')
    return lines


def gen_image_from_dcm(dcm_path, min_max_values):
    # print(dcm_path, type(dcm_path))
    img16_raw = numpy.float32(numpy.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(dcm_path)))))
    imgs = []
    for min_value, max_value in min_max_values:
        img16 = img16_raw.copy()
        min_value = img16.min() if min_value == 'min' else min_value
        max_value = img16.max() if max_value == 'max' else max_value
        img16[img16 > max_value] = max_value
        img16[img16 < min_value] = min_value
        img8u = numpy.uint8((img16 - min_value) * 255. / (max_value - min_value))
        imgs.append(img8u)
    return imgs


def get_rounded_pts(pts_list, index_range, stride=3.0, as_unique=False):
    s, e = index_range
    pts = ((numpy.array(pts_list)[s:e] / stride).round() * stride).astype(int)
    return numpy.unique(pts, axis=0) if as_unique else pts


def map_roi_to_3d(roi, c2d_pts, c3d_pts, stride=3.0):
    roi_pts = numpy.float32(roi['edge'])
    return map_pts2d_to_3d(roi_pts, c2d_pts, c3d_pts, stride)


def map_rect_to_3d(rect, c2d_pts, c3d_pts, stride=3.0):
    x, y, w, h = rect
    pts = []
    for i in range(int(math.ceil(x)), int(math.ceil(x + w))):
        for j in range(int(math.ceil(y)), int(math.ceil(y + h))):
            pts.append([i, j])
    if len(pts) > 50:
        pts = random.sample(pts, 50)
    pts2d = numpy.float32(pts + [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return map_pts2d_to_3d(pts2d, c2d_pts, c3d_pts, stride)


def map_pts2d_to_3d(pts2d, c2d_pts, c3d_pts, stride=3.0):
    d2 = pairwise_dist2(pts2d, c2d_pts)
    min_dist_indices = d2.argmin(axis=1)
    min_idx, max_idx = min_dist_indices.min(), min_dist_indices.max()
    min_dists = d2.min(axis=1)
    #print(d2.shape, min_dist_indices.shape, min_idx, max_idx, c2d_pts[min_idx], c2d_pts[max_idx], min_dists.min(), min_dists.max())
    #print(c3d_pts[min_idx], c3d_pts[max_idx])
    if min_dists.min() > 100:
        return None
    return get_rounded_pts(c3d_pts, [min_idx, max_idx + 1], stride, as_unique=True)


def cl_pts_iou(cl1, cl2, criteria="min"):
    cl = numpy.concatenate([cl1, cl2])
    union_num = len(numpy.unique(cl, axis=0))
    inter_num = len(cl1) + len(cl2) - union_num
    if criteria == 'min':
        return inter_num * 1. / (min(len(cl1), len(cl2)) + 1e-5)
    return inter_num * 1. / (union_num + 1e-5)


def get_c3d_merged_groups(cl_list, ratio=0.25, criteria="min"):
    N = len(cl_list)
    assert criteria == "min"
    id_pairs = []
    for i in range(N):
        for j in range(N):
            if i > j:
                continue
            if i == j:
                id_pairs.append([i, j])
            cl1, cl2 = cl_list[i], cl_list[j]
            divisor = min(len(cl1), len(cl2))
            cl = numpy.concatenate([cl1, cl2])
            divident = len(cl1) + len(cl2) - len(numpy.unique(cl, axis=0))
            overlap = divident * 1.0 / divisor
            if overlap > ratio:
                id_pairs.append([i, j])
    uf = UnionFind(id_pairs)
    return uf.run()


class Plaque3D(object):
    def __init__(self, pts3d=None, les_type=None, size=None, roi_num=1, stenosis='none', score=0):
        self.pts3d = pts3d
        self.les_type = les_type
        self.size = size
        self.roi_num = roi_num
        self.stenosis = stenosis
        self.score = score

    def to_json(self):
        d = {
            'pts3d': self.pts3d.tolist(),
            'les_type': self.les_type,
            'size': self.size,
            'roi_num': self.roi_num,
            'stenosis': self.stenosis,
            'score': self.score
        }
        return d

    def from_json(self, json):
        self.pts3d = numpy.float32(json['pts3d'])
        self.les_type = json['les_type']
        self.size = json['size']
        self.roi_num = json['roi_num']
        self.stenosis = json['stenosis']
        self.score = json.get('score', 0)

    def get_center(self):
        return self.pts3d.mean(axis=0)

    @staticmethod
    def merge(p3d_list):
        merged_p3d_list = []
        plaque_list, stent_list, xjq_list = [], [], []
        for p3d in p3d_list:
            if p3d.les_type == 'stent':
                stent_list.append(p3d)
            elif p3d.les_type == 'xjq':
                xjq_list.append(p3d)
            else:
                plaque_list.append(p3d)
        for sub_p3d_list in [plaque_list, stent_list, xjq_list]:
            cl_list = [p3d.pts3d for p3d in sub_p3d_list]
            group_ids = get_c3d_merged_groups(cl_list)
            for indices in group_ids:
                les_types = list(set([sub_p3d_list[i].les_type for i in indices]))
                merged_les_type = les_types[0] if len(les_types) == 1 else 'mix'
                merged_size = max([sub_p3d_list[i].size for i in indices])
                stenosis_set = set([sub_p3d_list[i].stenosis for i in indices])
                merged_stenosis = 'none'
                for s in g_stenosis_type_list[::-1]:
                    if s in stenosis_set:
                        merged_stenosis = s
                        break
                merged_score = sum([sub_p3d_list[i].score for i in indices])
                merged_pts3d = numpy.unique(numpy.concatenate([sub_p3d_list[i].pts3d for i in indices]), axis=0)
                merged_p3d_list.append(
                    Plaque3D(merged_pts3d, merged_les_type, merged_size, len(indices), merged_stenosis, merged_score))
        return merged_p3d_list


def get_cta_pixel_spacings(cta_dcm_dir):
    dcm_paths = glob.glob(cta_dcm_dir + '/*.dcm')
    dcm1 = dicom_load(dcm_paths[0])
    dcm2 = dicom_load(dcm_paths[1])
    dz = float(dcm1.ImagePositionPatient[2]) - float(dcm2.ImagePositionPatient[2])
    di = int(dcm1.InstanceNumber) - int(dcm2.InstanceNumber)
    spacing_x, spacing_y = float(dcm1.PixelSpacing[0]), float(dcm1.PixelSpacing[1])
    spacing_z = abs(dz / di)
    return spacing_x, spacing_y, spacing_z
