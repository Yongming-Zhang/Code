from cta_util import *
import os
import glob
from lib.utils.mio import *


def soft_link_safe(src_path, dst_path, is_cover=False):
    if os.path.exists(dst_path):
        if is_cover:
            os.remove(dst_path)
        else:
            return
    try:
        os.symlink(src_path, dst_path)
    except Exception as e:
        os.remove(dst_path)
        os.symlink(src_path, dst_path)


def gen_one_data(src_dir, json_dir, dst_dir, psid):
    dcm_paths = glob.glob(src_dir + psid + '/*/*.dcm')
    dst_base_dir = dst_dir + psid + '_CTA/'
    mkdir_safe(dst_base_dir)
    dst_cpr_dcm_dir = dst_base_dir + 'cprCoronary/'
    mkdir_safe(dst_cpr_dcm_dir)
    soft_link_safe(src_dir + psid + '/centerline/', dst_base_dir + 'centerline')
    map_coro_idx_dict = {}
    for dcm_path in dcm_paths:
        dcm_name = dcm_path.split('/')[-1]
        vessel_name = dcm_name.split('_')[0]
        map_coro_idx_dict[vessel_name] = vessel_name
        dst_path = dst_cpr_dcm_dir + dcm_name
        soft_link_safe(dcm_path, dst_path)
        soft_link_safe(dcm_path + '.bmp', dst_path + '.bmp')
    json_save(dst_base_dir + 'map_coro_idx.txt', map_coro_idx_dict)
    center_line_json = json_load(json_dir + psid + '.json')
    result_json = {
        'instance_number_start': 1,
        'patientID': psid.split('_')[0],
        'studyUID': psid.split('_')[1],
        'seriesUID': '',
        'pixel_spacing': 0.5,
        'slice_thickness': 0.5,
        'slice_spacing': 0.5,
        'center_lines': []
    }
    for c in center_line_json:
        c['vessel_id'] = c['name']
        result_json['center_lines'].append(c)
    json_save(dst_base_dir + 'result.json', result_json)


def gen_data():
    base_dir = '/breast_data/cta/new_data/b1_bt_587/'
    src_dir = base_dir + 'cpr_s18_n20_reorg_dcm_bmp/'
    json_dir = base_dir + 'centerline_json/'
    dst_dir = base_dir + 'data_for_drwise/'
    psid_list = load_string_list(base_dir + 'config/psid_list.txt')
    for psid in psid_list:
        if os.path.exists(src_dir + psid):
            gen_one_data(src_dir, json_dir, dst_dir, psid)


def link_cta_dicom():
    base_dir = '/breast_data/cta/new_data/b1_bt_587/'
    dcm_dir = base_dir + 'cta_dicom/'
    dst_dir = base_dir + 'data_for_drwise/'
    for ps_dcm_dir in sorted(glob.glob(dcm_dir + '*/')):
        soft_link_safe(ps_dcm_dir, dst_dir + ps_dcm_dir.split('/')[-2], is_cover=True)


if __name__ == '__main__':
    # gen_data()
    link_cta_dicom()
