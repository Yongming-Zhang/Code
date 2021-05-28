from cta_util import *
from collections import namedtuple, defaultdict, Counter
from cta.cta_json import gen_cta_psid_json_dict, get_cta_nodes
from lib.utils.mio import json_save, mkdir_safe, load_string_list


C3dMapper = namedtuple('C3dMapper', 'pts3d name c3d dist2_mat c3d_nearest_indices c3d_nearest_dist2s')


def get_c3d_mapper(pts3d, vessel_name, vessel_c3d):
    d2_mat = pairwise_dist2(pts3d, vessel_c3d)
    c3d_nearest_indices = d2_mat.argmin(axis=1)
    c3d_nearest_dist2s = d2_mat.min(axis=1)
    mapper = C3dMapper(
        pts3d=pts3d, name=vessel_name, c3d=vessel_c3d,
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


def get_centerline_range_len(c3d, i0, i1, pixel_spacings):
    pts3d = c3d[i0:(i1+1):2] if i1 - i0 >= 3 else c3d[i0:(i1+1)]
    if pts3d.shape[0] == 1:
        return 1
    pts3d_delta = pts3d[:-1] - pts3d[1:]
    pts3d_delta_mm = numpy.float32([pixel_spacings] * pts3d_delta.shape[0]) * pts3d_delta
    dist_accum = numpy.sqrt(numpy.sum(pts3d_delta_mm ** 2, axis=1)).sum()
    return dist_accum / pixel_spacings[0]


def gen_node_p3d(cta_node, vessel_c3d_dict, pixel_spacings):
    valid_les_types = {
        'low': 'low', 'cal': 'cal', 'mix': 'mix', 'clot': 'block', 'low_density': 'low', 'calc': 'cal',
        'stent_clear': 'stent', 'stent_unclear': 'stent', 'stent_block': 'stent', 'stent_none': 'stent',
        'w_stent': 'stent', 'b_stent': 'stent', 'c_stent': 'stent', 'mb_mca': 'xjq', 'mca_mb': 'xjq'}
    if cta_node['type'] not in valid_les_types:
        return None
    les_type = valid_les_types[cta_node['type']]
    node_pts3d = numpy.float32(cta_node['pts3d'])
    c3d_mappers = [get_c3d_mapper(node_pts3d, name, c3d) for name, c3d in vessel_c3d_dict.items()]
    best_c3d_mapper = find_best_vessel(c3d_mappers)
    if best_c3d_mapper is None:
        return None
    i0 = best_c3d_mapper.c3d_nearest_indices.min()
    i1 = best_c3d_mapper.c3d_nearest_indices.max()
    pts3d = get_rounded_pts(best_c3d_mapper.c3d, [i0, i1 + 1], 3, as_unique=True)
    cl_len = get_centerline_range_len(best_c3d_mapper.c3d, i0, i1, pixel_spacings)
    p3d = Plaque3D(pts3d, les_type, size=cl_len, stenosis=cta_node['stenosis'])
    # print(node_pts3d)
    # print(cta_node['segments'], best_c3d_mapper.name)
    # print(best_c3d_mapper.c3d_nearest_indices)
    # print(i0, i1, pts3d.shape, cl_len, les_type)
    # print('')
    return p3d


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
        # cta_json_dict_list = json_load(base_dir + 'annotation/task_1265_0-1588272710.json')
        # cta_json_dict_list = json_load(base_dir + 'annotation/task_1744_0-1597827901.json')
        # cta_json_dict_list += json_load(base_dir + 'annotation/task_1772_0-1600121101.json')
        cta_json_dict_list = json_load(base_dir + 'annotation/plaque_check/task_1744_0-1597827901.json')
        cta_json_dict_list += json_load(base_dir + 'annotation/plaque_check/task_1772_0-1602303384.json')
        cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    else:
        base_dir = '/breast_data/cta/new_data/b3_azcto_150/'
        cta_json_dict_list = json_load(base_dir + 'annotation/task_1830_0-1599383101.json')
        cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list, True)
    return base_dir, cta_psid_json_dicts


def load_vessel_c3d_dict(centerline_dir):
    c3d_paths = glob.glob(centerline_dir + '*.3d')
    vessel_name_map = load_vessel_name_map()
    vessel_c3d_dict = {}
    for c3d_path in c3d_paths:
        c3d = json_load(c3d_path)
        name = vessel_name_map.get(c3d['name'], c3d['name'])
        vessel_c3d_dict[name] = numpy.float32(c3d['points'])
    return vessel_c3d_dict


def gen_cta_pts3d(cta_psid_json_dicts, psid_list, base_dir):
    cpr_lumen_dir = base_dir + 'raw/cpr_lumen_s9_n20/'
    cta_dcm_dir = base_dir + 'cta_dicom/'
    dst_ann_dir = base_dir + 'annotation/cta/p3d_s3_checked_json/'
    mkdir_safe(dst_ann_dir)
    for psid in psid_list:
        if psid not in cta_psid_json_dicts:
            continue
        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + psid + '/*.dcm'))
        pixel_spacings = get_cta_pixel_spacings(cta_dcm_dir + psid + '/')
        cta_instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(cta_dcm_paths)}
        cta_ann_nodes = get_cta_nodes(cta_psid_json_dicts[psid], cta_instance_dict)
        vessel_c3d_dict = load_vessel_c3d_dict(cpr_lumen_dir + psid + '/cpr/centerline/')
        p3d_list = []
        for node in cta_ann_nodes:
            p3d = gen_node_p3d(node, vessel_c3d_dict, pixel_spacings)
            if p3d is not None:
                p3d_list.append(p3d)
        p3d_types = [p3d.les_type for p3d in p3d_list]
        p3d_json_list = [p3d.to_json() for p3d in p3d_list]
        json_save(dst_ann_dir + psid + '.json', p3d_json_list)
        print('processed %s: %d nodes, %d p3d, types: %s' % (psid, len(cta_ann_nodes), len(p3d_json_list), p3d_types))


def gen_cta_pts3d_main():
    for dataset_name in ['b2']:
        base_dir, cta_psid_json_dicts = load_dataset(dataset_name)
        for set_name in ['valid', 'test']:
            psid_list = load_string_list(base_dir + 'config/%s_psid_list.txt' % set_name)
            gen_cta_pts3d(cta_psid_json_dicts, psid_list, base_dir)


if __name__ == '__main__':
    gen_cta_pts3d_main()

