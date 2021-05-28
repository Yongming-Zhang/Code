from cta_util import *
from collections import namedtuple, defaultdict, Counter
from cta.cta_json import gen_cta_psid_json_dict, get_cta_nodes


def load_blind_jd():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    cta_json_dict_list = json_load(base_dir + 'annotation/task_1265_0-1588272710.json')
    cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    return cta_psid_json_dicts


def load_check_jd():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    cta_json_dict_list = json_load(base_dir + 'annotation/task_1744_0-1597827901.json')
    cta_json_dict_list += json_load(base_dir + 'annotation/task_1772_0-1600121101.json')
    cta_psid_json_dicts = gen_cta_psid_json_dict(cta_json_dict_list)
    return cta_psid_json_dicts


def cal_pts_iou(blind_nodes, check_nodes):
    blind_pts_list = [numpy.unique(node['pts3d'], axis=0) for node in blind_nodes]
    check_pts_list = [numpy.unique(node['pts3d'], axis=0) for node in check_nodes]
    iou_mat = numpy.zeros([len(blind_pts_list), len(check_pts_list)], dtype='float32')
    for b, bpts in enumerate(blind_pts_list):
        for c, cpts in enumerate(check_pts_list):
            iou_mat[b, c] = cl_pts_iou(bpts, cpts, 'min')
    return iou_mat


def count_confusion_matrix(check_blind_type_iou_pairs):
    cm_table = defaultdict(lambda: defaultdict(lambda: 0))
    check_prefix = 'check_'
    for ctype, btype, iou in check_blind_type_iou_pairs:
        for ct in ['total', ctype]:
            for bt in ['total', btype]:
                cm_table[check_prefix + ct][bt] += 1
    recall_str_table = defaultdict(lambda: defaultdict(lambda: '0'))
    types = ['fp', 'cal', 'low', 'mix', 'total']
    row_names = [check_prefix + t for t in types]
    for row_name in row_names:
        for col_name in types:
            recall_str_table[row_name][col_name] = fmt_ratio_str(
                cm_table[row_name][col_name], cm_table[row_name]['total'])
    lines = gen_table_strings(
        recall_str_table, '', align='c', col_unit_len=17, row_names=row_names, col_names=types)
    for line in lines:
        print('%s' % line)


def compare_plaque():
    blind_json_dicts = load_blind_jd()
    check_json_dicts = load_check_jd()
    cta_dcm_dir = '/breast_data/cta/new_data/b2_sg_407/cta_dicom/'
    check_blind_type_iou_pairs = []
    iou_thresh = 0.1
    for psid, blind_jd in blind_json_dicts.items():
        if psid not in check_json_dicts:
            continue
        blind_doc_id = blind_jd['other_info']['doctorId']
        cta_dcm_paths = sorted(glob.glob(cta_dcm_dir + psid + '/*.dcm'))
        cta_instance_dict = {int(p.split('/')[-1][:-4]): i for i, p in enumerate(cta_dcm_paths)}
        blind_nodes = get_cta_nodes(blind_jd, cta_instance_dict, max_pts_num=-1)
        check_nodes = get_cta_nodes(check_json_dicts[psid], cta_instance_dict, max_pts_num=-1)
        iou_mat = cal_pts_iou(blind_nodes, check_nodes)
        cur_check_blind_type_iou_pairs = []
        for c, cnode in enumerate(check_nodes):
            if cnode['type'] not in ['cal', 'low', 'mix']:
                continue
            iou = 0 if len(blind_nodes) == 0 else iou_mat[:, c].max()
            btype = blind_nodes[iou_mat[:, c].argmax()]['type'] if iou >= iou_thresh else 'fp'
            cur_check_blind_type_iou_pairs.append([cnode['type'], btype, '%.3f' % iou])
        for b, bnode in enumerate(blind_nodes):
            if bnode['type'] not in ['cal', 'low', 'mix']:
                continue
            iou = iou_mat[b, :].max() if len(check_nodes) > 0 else 0
            if iou < iou_thresh:
                cur_check_blind_type_iou_pairs.append(['fp', bnode['type'], '%.3f' % iou])
        print('%s: %s' % (psid, cur_check_blind_type_iou_pairs))
        check_blind_type_iou_pairs += cur_check_blind_type_iou_pairs
    count_confusion_matrix(check_blind_type_iou_pairs)


if __name__ == '__main__':
    compare_plaque()

