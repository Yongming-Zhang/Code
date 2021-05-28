from cta_util import *
import pydicom
from lib.utils.mio import save_string_list, load_string_list, json_save


def gen_cpr_instance_number_map(src_dir):
    dcm_paths = sorted(glob.glob(src_dir + '*/*/*.dcm'))
    print('%d dcm files' % len(dcm_paths))
    lines = []
    for dcm_path in dcm_paths:
        dcm = pydicom.dcmread(dcm_path)
        instance = dcm.InstanceNumber
        psid, vessel_name, file_name = dcm_path.split('/')[-3:]
        lines.append(psid + ' ' + vessel_name + ' ' + file_name + ' ' + str(instance))
    save_string_list(src_dir + 'instance_numbers.txt', lines)


def load_psvd_instance_dict(path):
    lines = load_string_list(path)
    d = {}
    for line in lines:
        psid, vessel_name, file_name, instance = line.split(' ')
        d[psid + '_' + vessel_name + '_' + file_name[:-4]] = int(instance)
    return d


def load_psvi_psvd_dict(path):
    lines = load_string_list(path)
    d = {}
    for line in lines:
        psid, vessel_name, file_name, instance = line.split(' ')
        psvd = psid + '_' + vessel_name + '_' + file_name[:-4]
        d[psid + '_' + vessel_name + '_' + instance] = psvd
    return d


def map_instance_numbers():
    base_dir = '/data2/zhangfd/data/cta/b2_407/cta2cpr/'
    src_map_file = base_dir + 'cpr_s9_n20_reorg/instance_numbers.txt'
    dst_map_file = base_dir + 'cpr_s9_n20_reorg/old_instance_numbers.txt'
    src_json_dir = base_dir + 'annotation/cpr_s9_n20_reorg/check_json/'
    dst_json_dir = base_dir + 'annotation/cpr_s9_n20_reorg/check_json_old_instance/'
    src_psvi_psvd_dict = load_psvi_psvd_dict(src_map_file)
    dst_psvd_instance_dict = load_psvd_instance_dict(dst_map_file)
    for json_path in sorted(glob.glob(src_json_dir + '*.json')):
        cpr_json_dict = json_load(json_path)
        psv = json_path.split('/')[-1][:-5]
        for node in cpr_json_dict['nodes']:
            if 'rois' not in node:
                continue
            for roi in node['rois']:
                psvi = psv + '_%d' % roi['slice_index']
                psvd = src_psvi_psvd_dict[psvi]
                instance = dst_psvd_instance_dict[psvd]
                roi['slice_index'] = instance
        print('%s: %d nodes' % (json_path, len(cpr_json_dict['nodes'])))
        json_save(dst_json_dir + psv + '.json', cpr_json_dict)


if __name__ == '__main__':
    # src_dir08 = '/data2/zhangfd/data/cta/b2_407/cta2cpr/cpr_s9_n20_reorg/'
    # src_dir_breast_data = '/breast_data/cta/new_data/b2_sg_407/cpr_s9_n20_reorg_dcm_bmp/'
    # gen_cpr_instance_number_map(src_dir_breast_data)
    map_instance_numbers()

