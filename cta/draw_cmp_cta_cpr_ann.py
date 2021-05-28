# -*- coding: utf-8 -*-
from cta_util import *
import cv2
import glob
import numpy
from collections import defaultdict
from lib.utils.mio import dicom_load, load_image, mkdir_safe, save_image, json_load
from cta_cpr_map import CtaCprMapper, load_cpr_coord_map
from lib.image.image import gray2rgb, pad_img
from lib.image.draw import draw_texts
from lib.utils.contour import find_contours, fill_contours, draw_contours


g_color_dict = {u'钙化': (0, 255, 0), u'低密度': (0, 0, 255), u'混合': (0, 97, 255), u'闭塞': (255, 0, 255)}
g_stenosis_dict = {
    u'无明显狭窄': '0%', u'轻微（1%-24%）': '1%-24%', u'轻度（25%-49%）': u'25%-49%',
    u'中度（50%-69%）': '50%-69%', u'重度（70%-99%）': '70%-99%', u'闭塞（100%）': '100%'
}


def map_cta_annotation_to_cpr(cpr_coord_maps, cta_ann_nodes, cta_shape, instance_off=1, ds2_node_slice_num=-1):
    mapper = CtaCprMapper(cpr_coord_maps)
    cpr_mapped_nodes_list = [[] for i in range(len(cpr_coord_maps))]
    total_roi_num = 0
    for node in cta_ann_nodes:
        les_type = node['descText'][1][0][0]['select']
        stenosis_cn_str = node['descText'][1][1][0]['select']
        if stenosis_cn_str not in g_stenosis_dict:
            continue
        stenosis = g_stenosis_dict[stenosis_cn_str]
        cta_mask = numpy.zeros(cta_shape, dtype='uint8')
        if les_type not in g_color_dict:
            continue
        slice_indices = []
        for roi in node['rois'] + node['bounds']:
            contours_numpy = numpy.int32(roi['edge'])
            z = roi['slice_index'] - instance_off
            slice_indices.append(z)
            cv2.drawContours(cta_mask[z], (contours_numpy,), 0, color=255, thickness=-1)
            total_roi_num += 1
        if len(set(slice_indices)) > ds2_node_slice_num > 0:
            cta_mask[::2] = cta_mask[1::2]
        cpr_masks = mapper.cta2cpr(cta_mask)
        for i, cpr_mask in enumerate(cpr_masks):
            edge_list = find_contours(cpr_mask)
            for edge in edge_list:
                node = {'edge': edge[:, 0, :], 'type': les_type, 'stenosis': stenosis}
                cpr_mapped_nodes_list[i].append(node)
    print('\tannotated %d rois in all' % total_roi_num)
    return cpr_mapped_nodes_list


def parse_pssi_nodes_dict(jd_list):
    pssi_nodes_dict = defaultdict(list)
    for jd in jd_list:
        pid, study, series = jd['patientID'].split('_')[0], jd['studyUID'], jd['seriesUID']
        for node in jd['nodes']:
            les_type = node['descText'][1][0][0]['select']
            if les_type not in g_color_dict:
                continue
            for roi in node['rois'] + node['bounds']:
                pssi = pid + '_' + study + '_%d' % roi['slice_index']
                pssi_nodes_dict[pssi].append({'edge': roi['edge'], 'type': les_type})
    return pssi_nodes_dict


def draw_title(img_rgb, title):
    draw_texts(img_rgb, (0, 0), [title], (255, 255, 0), is_bold=True, direct=-1, font_scale=0.7)


def draw_nodes(img, nodes):
    for node in nodes:
        color = g_color_dict[node['type']]
        edge_numpy = numpy.int32(node['edge'])
        fill_contours(img, edge_numpy, color)
        # if 'stenosis' in node:
        #     x0, y0 = edge_numpy.min(axis=0)
        #     draw_texts(img, [x0, y0], [node['stenosis']], color, font_scale=0.5)


def draw_and_cmp_one_cpr(cpr_image, cpr_image_old, cta_doc, cpr_cta_nodes, cpr_checked_nodes, cpr_check_doc, cpr_blind_nodes, cpr_blind_doc):
    img_h = max(cpr_image.shape[0], cpr_image_old.shape[0])
    img_w = max(cpr_image.shape[1], cpr_image_old.shape[1])
    cpr_image = pad_img(cpr_image, [0, img_w - cpr_image.shape[1], 0, img_h - cpr_image.shape[0]])
    cpr_image_old = pad_img(cpr_image_old, [0, img_w - cpr_image_old.shape[1], 0, img_h - cpr_image_old.shape[0]])
    img_rgb = gray2rgb(cpr_image)
    img_cta_ann_rgb = gray2rgb(cpr_image)
    draw_title(img_cta_ann_rgb, 'From CTA, Doc=%s' % cta_doc)
    draw_nodes(img_cta_ann_rgb, cpr_cta_nodes)
    new_img_row = numpy.hstack([img_rgb, img_cta_ann_rgb])
    img_checked_rgb = gray2rgb(cpr_image_old)
    draw_title(img_checked_rgb, 'CPR Check, Doc=%s' % cpr_check_doc)
    draw_nodes(img_checked_rgb, cpr_checked_nodes)
    img_blind_rgb = gray2rgb(cpr_image_old)
    draw_title(img_blind_rgb, 'CPR Blind, Doc=%s' % cpr_blind_doc)
    draw_nodes(img_blind_rgb, cpr_blind_nodes)
    old_img_row = numpy.hstack([img_checked_rgb, img_blind_rgb])
    # img_w = min(new_img_row.shape[1], old_img_row.shape[1])
    return numpy.vstack([new_img_row, old_img_row])


def draw_and_cmp(cta_ann_jd, cta_dcm_base_dir, cpr_ann_jds, cpr_checked_ann_jds, cpr_image_base_dir, cpr_info_base_dir, base_save_dir):
    pid, study = cta_ann_jd['patientID'], cta_ann_jd['studyUID']
    cpr_dcm_paths = glob.glob(cpr_info_base_dir + pid + '_' + study + '/cpr/*.dcm')
    cpr_coord_maps = [load_cpr_coord_map(dcm_path + '.dat') for dcm_path in cpr_dcm_paths]
    cta_dcm_paths = glob.glob(cta_dcm_base_dir + pid + '_' + study + '/*.dcm')
    cta_dcm_paths.sort()
    print('%s_%s: %d cta_dcm, %d cta nodes, %d cpr ann json, %d cpr check json' % (
        pid, study, len(cta_dcm_paths), len(cta_ann_jd['nodes']), len(cpr_ann_jds), len(cpr_checked_ann_jds)))
    cta_shape = [len(cta_dcm_paths), 512, 512]
    instance_off = int(cta_dcm_paths[0].split('/')[-1][:-4])
    cpr_cta_mapped_nodes_list = map_cta_annotation_to_cpr(cpr_coord_maps, cta_ann_jd['nodes'], cta_shape, instance_off)
    cpr_blind_pssi_nodes_dict = parse_pssi_nodes_dict(cpr_ann_jds)
    cpr_checked_pssi_nodes_dict = parse_pssi_nodes_dict(cpr_checked_ann_jds)
    for cpr_cta_mapped_nodes, cpr_dcm_path in zip(cpr_cta_mapped_nodes_list, cpr_dcm_paths):
        # img16 = numpy.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(dcm_path)))
        dcm = dicom_load(cpr_dcm_path)
        instance_num = str(dcm.InstanceNumber)
        file_name = cpr_dcm_path.split('/')[-1][:-4]
        series_desc, degree = file_name.split('_')
        image_path_old = cpr_image_base_dir + pid + '_' + study + '_' + series_desc + '_' + instance_num + '.png'
        pssi = pid + '_' + study + '_' + instance_num
        cta_node_num = len(cpr_cta_mapped_nodes)
        cpr_node_check_num = len(cpr_checked_pssi_nodes_dict[pssi])
        cpr_node_blind_num = len(cpr_blind_pssi_nodes_dict[pssi])
        print('%s_%s/%s, %d nodes from CTA, %d checked %d blind from CPR' % (
            pid, study, file_name, cta_node_num, cpr_node_check_num, cpr_node_blind_num))
        if cta_node_num > 0 or cpr_node_check_num > 0 or cpr_node_blind_num > 0:
            save_dir = base_save_dir + pid + '_' + study + '/'
            mkdir_safe(save_dir)
            image = load_image(cpr_info_base_dir + pid + '_' + study + '/image/range_-100_700/' + file_name + '.png')
            image_old = load_image(image_path_old)
            draw_image = draw_and_cmp_one_cpr(
                image,
                image_old,
                g_doctor_id_names[cta_ann_jd['other_info']['doctorId']],
                cpr_cta_mapped_nodes,
                cpr_checked_pssi_nodes_dict[pssi],
                g_doctor_id_names[cpr_checked_ann_jds[0]['other_info']['doctorId']],
                cpr_blind_pssi_nodes_dict[pssi],
                g_doctor_id_names[cpr_ann_jds[0]['other_info']['doctorId']],
            )
            save_image(save_dir + file_name + '.png', draw_image)


def parse_cpr_json_dicts(json_list):
    check_ps_json_dicts = defaultdict(list)
    blind_ps_json_dicts = defaultdict(list)
    for jd in json_list:
        if jd['other_info']['passReason'] != '':
            continue
        pid, study = jd['patientID'].split('_')[0], jd['studyUID']
        if jd['other_info']['isReview'] != 0:
            check_ps_json_dicts[pid + '_' + study].append(jd)
        else:
            blind_ps_json_dicts[pid + '_' + study].append(jd)
    return check_ps_json_dicts, blind_ps_json_dicts


def draw_main():
    base_dir = '/data2/zhangfd/data/cta/b1_587/cta2cpr/'
    base_save_dir = base_dir + 'draw/cmp_cta_cpr/'
    cta_ann_jds = json_load(base_dir + 'annotation/task_10816_0-1585490702.json')
    cpr_ann_jds = json_load(base_dir + 'annotation/task_10627_0-1583319936.json')
    check_ps_cpr_ann_jds, blind_ps_cpr_ann_jds = parse_cpr_json_dicts(cpr_ann_jds)
    cta_dcm_base_dir = base_dir + 'cta_dicom/'
    cpr_image_base_dir = base_dir + 'image/range_-100_700/'
    cpr_info_base_dir = base_dir + 'cpr/'
    for cta_ann_jd in cta_ann_jds:
        i0, i1 = cta_ann_jd['other_info']['instance_range']
        if i1 - i0 < 10:
            continue
        if cta_ann_jd['other_info']['passReason'] != '':
            continue
        ps = cta_ann_jd['patientID'] + '_' + cta_ann_jd['studyUID']
        draw_and_cmp(cta_ann_jd, cta_dcm_base_dir, blind_ps_cpr_ann_jds[ps], check_ps_cpr_ann_jds[ps],
                     cpr_image_base_dir, cpr_info_base_dir, base_save_dir)


# ------------------------- Draw and compare origin and down sampled by 2 ------------------------- #
g_max_slice_num_no_ds = 10


def draw_and_ds2_one(cpr_image, cta_doc, cpr_cta_nodes, cpr_cta_nodes_ds2):
    img_rgb = gray2rgb(cpr_image)
    img_cta_ann_rgb = gray2rgb(cpr_image)
    # numpy.save('tmp.npy', img_cta_ann_rgb)
    print(img_cta_ann_rgb.dtype)
    draw_title(img_cta_ann_rgb, 'From CTA, Doc=%s' % cta_doc)
    draw_nodes(img_cta_ann_rgb, cpr_cta_nodes)
    img_cta_ann_ds2_rgb = gray2rgb(cpr_image)
    draw_title(img_cta_ann_ds2_rgb, 'From CTA, Doc=%s, DS/2-MAX%d' % (cta_doc, g_max_slice_num_no_ds))
    draw_nodes(img_cta_ann_ds2_rgb, cpr_cta_nodes_ds2)
    return numpy.vstack([img_rgb, img_cta_ann_rgb, img_cta_ann_ds2_rgb])


def draw_and_ds2(cta_ann_jd, cta_dcm_base_dir, cpr_info_base_dir, base_save_dir):
    pid, study = cta_ann_jd['patientID'], cta_ann_jd['studyUID']
    cpr_dcm_paths = glob.glob(cpr_info_base_dir + pid + '_' + study + '/cpr/*.dcm')
    cpr_coord_maps = [load_cpr_coord_map(dcm_path + '.dat') for dcm_path in cpr_dcm_paths]
    cta_dcm_paths = glob.glob(cta_dcm_base_dir + pid + '_' + study + '/*.dcm')
    cta_dcm_paths.sort()
    print('%s_%s: %d cta_dcm, %d cta nodes' % (pid, study, len(cta_dcm_paths), len(cta_ann_jd['nodes'])))
    cta_shape = [len(cta_dcm_paths), 512, 512]
    instance_off = int(cta_dcm_paths[0].split('/')[-1][:-4])
    # cpr_cta_mapped_nodes_list, cpr_cta_ds2_mapped_nodes_list = [[]] * len(cpr_dcm_paths), [[]] * len(cpr_dcm_paths)
    cpr_cta_mapped_nodes_list = map_cta_annotation_to_cpr(cpr_coord_maps, cta_ann_jd['nodes'], cta_shape, instance_off)
    cpr_cta_ds2_mapped_nodes_list = map_cta_annotation_to_cpr(
        cpr_coord_maps, cta_ann_jd['nodes'], cta_shape, instance_off, g_max_slice_num_no_ds)
    for cpr_cta_mapped_nodes, cpr_cta_ds2_mapped_nodes, cpr_dcm_path in zip(
            cpr_cta_mapped_nodes_list, cpr_cta_ds2_mapped_nodes_list, cpr_dcm_paths):
        file_name = cpr_dcm_path.split('/')[-1][:-4]
        cta_node_num = len(cpr_cta_mapped_nodes)
        print('%s_%s/%s, %d nodes from CTA' % (pid, study, file_name, cta_node_num))
        if cta_node_num >= 0:
            save_dir = base_save_dir + pid + '_' + study + '/'
            mkdir_safe(save_dir)
            image = load_image(cpr_info_base_dir + pid + '_' + study + '/image/range_-100_700/' + file_name + '.png')
            doc_name = g_doctor_id_names[cta_ann_jd['other_info']['doctorId']]
            draw_image = draw_and_ds2_one(image, doc_name, cpr_cta_mapped_nodes, cpr_cta_ds2_mapped_nodes)
            save_image(save_dir + file_name + '.png', draw_image)


def draw_ds2_main():
    base_dir = '/data2/zhangfd/data/cta/b2_407/cta2cpr/'
    base_save_dir = base_dir + 'draw/cmp_cta_cpr/'
    cta_ann_jds = json_load(base_dir + 'annotation/task_10818_0-1586415901.json')
    cta_dcm_base_dir = base_dir + 'cta_dicom/'
    cpr_info_base_dir = base_dir + 'cpr_s18/'
    for cta_ann_jd in cta_ann_jds:
        i0, i1 = cta_ann_jd['other_info']['instance_range']
        if i1 - i0 < 10:
            continue
        if cta_ann_jd['other_info']['passReason'] != '':
            continue
        draw_and_ds2(cta_ann_jd, cta_dcm_base_dir, cpr_info_base_dir, base_save_dir)


def draw_check_blind_one(cpr_image_old, cpr_checked_nodes, cpr_check_doc, cpr_blind_nodes, cpr_blind_doc):
    img_rgb = gray2rgb(cpr_image_old)
    img_checked_rgb = gray2rgb(cpr_image_old)
    draw_title(img_checked_rgb, 'CPR Check, Doc=%s' % cpr_check_doc)
    draw_nodes(img_checked_rgb, cpr_checked_nodes)
    img_blind_rgb = gray2rgb(cpr_image_old)
    draw_title(img_blind_rgb, 'CPR Blind, Doc=%s' % cpr_blind_doc)
    draw_nodes(img_blind_rgb, cpr_blind_nodes)
    return numpy.vstack([img_rgb, img_checked_rgb, img_blind_rgb])


def draw_check_blind(cpr_ann_jds, cpr_checked_ann_jds, cpr_image_base_dir, base_save_dir, ps):
    pid, study = ps.split('_')
    print('%s_%s: %d cpr ann json, %d cpr check json' % (pid, study, len(cpr_ann_jds), len(cpr_checked_ann_jds)))
    cpr_blind_pssi_nodes_dict = parse_pssi_nodes_dict(cpr_ann_jds)
    cpr_checked_pssi_nodes_dict = parse_pssi_nodes_dict(cpr_checked_ann_jds)
    pssi_list = sorted(list(set(cpr_blind_pssi_nodes_dict.keys() + cpr_checked_pssi_nodes_dict.keys())))
    for pssi in pssi_list:
        pid, study, instance_num = pssi.split('_')
        image_path_old = glob.glob(cpr_image_base_dir + pid + '_' + study + '_*_' + instance_num + '.png')[0]
        cpr_node_check_num = len(cpr_checked_pssi_nodes_dict[pssi])
        cpr_node_blind_num = len(cpr_blind_pssi_nodes_dict[pssi])
        series_desc = image_path_old.split('/')[-1][:-4].split('_')[-2]
        print('%s, %d checked %d blind from CPR' % (pssi, cpr_node_check_num, cpr_node_blind_num))
        if cpr_node_check_num > 0 or cpr_node_blind_num > 0:
            save_dir = base_save_dir + pid + '_' + study + '/'
            mkdir_safe(save_dir)
            image_old = load_image(image_path_old)
            draw_image = draw_check_blind_one(
                image_old,
                cpr_checked_pssi_nodes_dict[pssi],
                g_doctor_id_names[cpr_checked_ann_jds[0]['other_info']['doctorId']],
                cpr_blind_pssi_nodes_dict[pssi],
                g_doctor_id_names[cpr_ann_jds[0]['other_info']['doctorId']],
            )
            save_image(save_dir + series_desc + '_' + instance_num + '.png', draw_image)


def draw_check_blind_main():
    base_dir = '/data2/zhangfd/data/cta/b1_587/cta2cpr/'
    cpr_image_base_dir = base_dir + 'image/range_-100_700/'
    dcm_src_dir = base_dir + 'test_ann_plat/test_cpr_cta_b2/'
    base_save_dir = dcm_src_dir + 'draw_check_blind2/'
    cpr_ann_jds = json_load(base_dir + 'annotation/task_10627_0-1583319936.json')
    cpr_ann_jds += json_load(base_dir + 'annotation/task_10652_0-1582016734.json')
    check_ps_cpr_ann_jds, blind_ps_cpr_ann_jds = parse_cpr_json_dicts(cpr_ann_jds)
    cpr_dcm_dirs = sorted(glob.glob(dcm_src_dir + 'cpr/*/'))
    for cpr_dcm_dir in cpr_dcm_dirs[14:]:
        ps = cpr_dcm_dir.split('/')[-2]
        draw_check_blind(blind_ps_cpr_ann_jds[ps], check_ps_cpr_ann_jds[ps], cpr_image_base_dir, base_save_dir, ps)


if __name__ == '__main__':
    # draw_main()
    # draw_ds2_main()
    draw_check_blind_main()

