from lumen_util import *
import math
from lib.utils.mio import *
from lib.utils.util import sample_list
from lib.utils.util import pairwise_dist2
from collections import defaultdict, namedtuple
from cta.cta_util import load_vessel_name_map, get_cta_pixel_spacings


Vessel = namedtuple('Vessel', 'name c3d lumen lids')
C3dMapper = namedtuple('C3dMapper', 'pts3d vessel dist2_mat c3d_nearest_indices c3d_nearest_dist2s')
label_dict = {
    'fp': 0,
    'low': 1,
    'cal': 2,
    'mix': 3,
    'stent': 4
}
label_name_dict = {v: k for k, v in label_dict.items()}
patch_size_xy, patch_size_z, patch_stride = 40, 40, 2
lumen_windows = [(300, 700), (-100, 700), (-500, 1000)]


def get_data_name():
    patch_str = '%dx%dx%d' % (patch_size_z, patch_size_xy, patch_size_xy)
    window_str = '_'.join(['%d-%d' % (x, y) for (x, y) in lumen_windows])
    return 'new_p%s_w%s' % (patch_str, window_str)


def crop_img(img, center_crop):
    img_h, img_w = img.shape
    x0, y0 = (img_w - center_crop) // 2, (img_h - center_crop) // 2
    img = img[y0:(y0 + center_crop), x0:(x0 + center_crop)]
    return img


def load_and_crop_lumen(lumen_dir):
    lumen_paths = sorted(glob.glob(lumen_dir + '*.dcm'))
    d = {}
    for lumen_path in lumen_paths:
        instance_num = int(lumen_path.split('/')[-1][:-4])
        imgs = gen_image_from_dcm(lumen_path, lumen_windows)
        rgb_img = numpy.zeros([patch_size_xy, patch_size_xy, 3], dtype='uint8')
        for c in range(3):
            rgb_img[:, :, c] = crop_img(imgs[c], patch_size_xy)
        d[instance_num] = rgb_img
    return d


def load_vessel(cpr_lumen_dir, spacing_x, spacing_y, spacing_z, vessel_id, vessel_name_map):
    centerline3d = json_load(cpr_lumen_dir + 'cpr/centerline/%s.3d' % vessel_id)
    c3d = centerline3d['points']
    lumen_img_dict = load_lumen_as_dict(cpr_lumen_dir + 'lumen/%s/' % vessel_id, patch_size_xy, patch_size_xy)
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


def get_c3d_mapper(pts3d, vessel):
    d2_mat = pairwise_dist2(pts3d, vessel.c3d)
    c3d_nearest_indices = d2_mat.argmin(axis=1)
    c3d_nearest_dist2s = d2_mat.min(axis=1)
    mapper = C3dMapper(
        pts3d=pts3d, vessel=vessel,
        dist2_mat=d2_mat, c3d_nearest_indices=c3d_nearest_indices, c3d_nearest_dist2s=c3d_nearest_dist2s)
    return mapper


def gen_gt_labels(json_dict):
    lumen_types = defaultdict(list)
    for node in json_dict['nodes']:
        node_type = node['type'].split('_')[0]
        if node_type == 'mca':
            continue
        for li in node['lumen_indices']:
            lumen_types[li].append(node_type)
    labels = {}
    for li, types in lumen_types.items():
        types = list(set(types))
        if len(types) == 1:
            merged_type = types[0]
        else:
            if 'stent' in types:
                merged_type = 'stent'
            else:
                merged_type = 'mix'
        labels[li] = label_dict[merged_type]
    return labels


def get_pred_lumen_indices(pred_pts, vessels):
    c3d_mappers = [get_c3d_mapper(pred_pts, vessel) for vessel in vessels]
    max_c3d_dist2, lumen_indices_list = 20, []
    for c3d_mapper in c3d_mappers:
        i0 = c3d_mapper.c3d_nearest_indices.min()
        i1 = c3d_mapper.c3d_nearest_indices.max()
        node_pts_nearest_dist2s = c3d_mapper.dist2_mat.min(axis=0)
        lumen_indices = []
        for i in range(i0, i1 + 1):
            if node_pts_nearest_dist2s[i] < max_c3d_dist2:
                li = c3d_mapper.vessel.lumen[i]
                lumen_indices.append(li)
        lumen_indices = sorted(list(set(lumen_indices)))
        lumen_indices_list.append(lumen_indices)
    return lumen_indices_list


def gen_one_patch(lumen_index, gt_label_dict, lumen_img_dict):
    l0 = lumen_index - patch_size_z // 2
    l1 = l0 + patch_size_z
    if lumen_index in gt_label_dict:
        label = gt_label_dict[lumen_index]
    else:
        z0, z1 = lumen_index - patch_size_z // 3, lumen_index + patch_size_z // 3
        labels = set([gt_label_dict.get(i, 0) for i in range(z0, z1)])
        label = 0 if len(labels) == 1 else -1
    if label < 0:
        return None, None
    empty_img = numpy.zeros([patch_size_xy, patch_size_xy, 3], dtype='uint8')
    imgs = [lumen_img_dict.get(i, empty_img) for i in range(l0, l1)]
    img_tensor = numpy.hstack(imgs)
    return img_tensor, label


def save_patch(lumen_index, prefix, gt_label_dict, lumen_img_dict, save_dir, vessel_name):
    if lumen_index % patch_stride > 0:
        return None
    img, label = gen_one_patch(lumen_index, gt_label_dict, lumen_img_dict)
    if img is None or label is None:
        return None
    save_dir_i = save_dir + label_name_dict[label] + '/'
    mkdir_safe(save_dir_i)
    img_path = save_dir_i + '%s_%s_lumen%d.png' % (prefix, vessel_name, lumen_index)
    save_image(img_path, img)
    line = '%s %dx%dx%d %d' % (img_path, patch_size_z, patch_size_xy, patch_size_xy, label)
    return line


def gen_patches(gt_json_dict, pred_pts_list, vessels, save_dir):
    gt_label_dict = gen_gt_labels(gt_json_dict)
    lines = []
    vessel_dict = {vessel.name: vessel for vessel in vessels}
    for pred_id, pred_pts in enumerate(pred_pts_list):
        pred_lumen_indices_list = get_pred_lumen_indices(pred_pts, vessels)
        for pred_lumen_indices, vessel in zip(pred_lumen_indices_list, vessels):
            for pred_li in pred_lumen_indices:
                line = save_patch(pred_li, 'P%d' % pred_id, gt_label_dict, vessel.lumen, save_dir, vessel.name)
                if line is not None:
                    lines.append(line)
    for gt_id, node in enumerate(gt_json_dict['nodes']):
        node_type = node['type'].split('_')[0]
        if node_type == 'mca':
            continue
        for vessel_name, lids in node['lumen_indices'].items():
            vessel = vessel_dict[vessel_name]
            for gt_li in lids:
                line = save_patch(gt_li, 'G%d' % gt_id, gt_label_dict, vessel.lumen, save_dir, vessel.name)
                if line is not None:
                    lines.append(line)
    return lines


def get_num_str(nums):
    return '/'.join(['l%d=%d' % (label, num) for label, num in enumerate(nums)])


def count_img_list(img_list, is_count_instance=False):
    nums = [0] * 5
    instance_nums = {}
    for line in img_list:
        path, pstr, label = line.split(' ')
        nums[int(label)] += 1
        if is_count_instance:
            name = path.split('/')[-1].split('_')[0]
            if name not in instance_nums:
                instance_nums[name] = [0] * 5
            instance_nums[name][int(label)] += 1
    count_str = get_num_str(nums)
    if is_count_instance:
        sub_strs = ['(%s: %s)' % (name, get_num_str(instance_nums[name])) for name in sorted(instance_nums.keys())]
        count_str += '\n  ' + ', '.join(sub_strs) + ''
    return count_str


def is_case_completed(sub_img_list_path):
    if not os.path.exists(sub_img_list_path):
        return False
    for line in load_string_list(sub_img_list_path):
        path, pstr, label = line.split(' ')
        if not os.path.exists(path):
            return False
    return True


def gen_set(base_dir, set_name):
    cta_dcm_dir = base_dir + 'cta_dicom/'
    model_base_dir = base_dir + 'model_data/'
    lumen_json_dir = base_dir + 'annotation/lumen_v2/json/'
    stage1_pred_dir = model_base_dir + 'stage1/cta_200608_cluster_mask_cas/' + set_name + '/'
    cpr_lumen_dir = base_dir + 'raw/cpr_lumen_s9_n20_reorg/'
    base_save_dir = model_base_dir + 'lumen_v2_patches_cls/'
    data_name = get_data_name()
    patch_save_dir = base_save_dir + 'patches/' + data_name + '/'
    list_save_dir = base_save_dir + 'lists/' + data_name + '/'
    mkdir_safe(list_save_dir + 'sub_lists/')
    pred_pts_list_dict = pickle_load(stage1_pred_dir + 'merged_pts3d.pkl')
    img_list = []
    pid_study_pairs = sorted(pred_pts_list_dict.keys())
    vessel_name_map = load_vessel_name_map()
    for no, (pid, study) in enumerate(pid_study_pairs):
        pred_pts_list = pred_pts_list_dict[(pid, study)]
        psid = pid + '_' + study
        gt_json_path = lumen_json_dir + psid + '.json'
        if not os.path.exists(gt_json_path):
            continue
        gt_json_dict = json_load(gt_json_path)
        spacing_x, spacing_y, spacing_z = get_cta_pixel_spacings(cta_dcm_dir + psid + '/')
        valid_vessels = None
        if base_dir == '/breast_data/cta/new_data/b3_azcto_150/':
            gen_json_paths = glob.glob(base_dir + 'annotation/cpr_s9_n20_reorg/raw_json/%s_*.json' % psid)
            valid_vessels = set([gen_json_path.split('_')[-1][:-5] for gen_json_path in gen_json_paths])
        vessels = load_all_vessels(
            cpr_lumen_dir + psid + '/', spacing_x, spacing_y, spacing_z, vessel_name_map, valid_vessels)
        sub_img_list_path = list_save_dir + 'sub_lists/' + psid + '.txt'
        if is_case_completed(sub_img_list_path):
            lines = load_string_list(sub_img_list_path)
        else:
            lines = gen_patches(gt_json_dict, pred_pts_list, vessels, patch_save_dir + psid + '/')
            save_string_list(sub_img_list_path, lines)
        img_list += lines
        save_string_list(list_save_dir + '%s_img_list.txt' % set_name, img_list)
        count_str = count_img_list(img_list) + '. ' + count_img_list(lines, True)
        pred_num, gt_num = len(pred_pts_list), len(gt_json_dict['nodes'])
        print('%s. No.%d/%d, %s: pred=%d, gt=%d, patch=%d, %s' % (
            set_name.upper(), no, len(pid_study_pairs), psid, pred_num, gt_num, len(lines), count_str))


def patch_main():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    gen_set(base_dir, 'train')
    gen_set(base_dir, 'valid')


def sample_one_list(img_list, list_dir, set_name):
    label_num_unit = 15000 if set_name == 'train' else 5000
    label_rates = [(1, 1, 1, 1, 1), (1, 2, 2, 2, 2), (2, 1, 1, 1, 1)]
    label_instance_lists = [defaultdict(list) for i in range(5)]
    origin_label_nums = [0] * 5
    for line in img_list:
        path, pstr, label = line.split(' ')
        psid, label_name, img_name = path.split('/')[-3:]
        instance_id = psid + '/' + label_name + '/' + img_name.split('_')[0]
        label_instance_lists[int(label)][instance_id].append(line)
        origin_label_nums[int(label)] += 1
    for label_rate in label_rates:
        new_img_list = []
        new_label_nums = [0] * 5
        for label, instance_lists in enumerate(label_instance_lists):
            new_label_patch_num = label_num_unit * label_rate[label]
            patch_per_instance = int(math.ceil(new_label_patch_num * 1. / len(instance_lists)))
            for instance_id, instance_img_list in instance_lists.items():
                new_img_list += sample_list(instance_img_list, patch_per_instance)
                new_label_nums[label] += patch_per_instance
        list_name = set_name + '_balance_label%s.txt' % ''.join([str(label) for label in label_rate])
        print('%s: %s, %d' % (list_name, str(new_label_nums), len(new_img_list)))
        save_string_list(list_dir + list_name, new_img_list)


def list_main():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    list_dir = base_dir + 'model_data/lumen/lists/' + get_data_name() + '/'
    for set_name in ['train', 'valid']:
        img_list = load_string_list(list_dir + '%s_img_list.txt' % set_name)
        sample_one_list(img_list, list_dir, set_name)


def parse_list():
    d = int(sys.argv[1])
    list_dir = '/breast_data/cta/new_data/b2_sg_407/model_data/lumen/lists/new_p%dx32x32_w300-700_-100-700_-500-1000/' % d
    list_path = list_dir + 'valid_img_list.txt'
    img_list = load_string_list(list_path)
    instance_label_vessel_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in img_list:
        path, pstr, label = line.split(' ')
        psid, label_name, img_name = path.split('/')[-3:]
        pred_id, vessel_name = img_name.split('_')[:2]
        if pred_id[0] == 'G':
            continue
        instance_id = psid + '/' + pred_id
        instance_label_vessel_list[instance_id][label_name][vessel_name].append(line)
    for instance_id in sorted(instance_label_vessel_list.keys()):
        label_vessel_list = instance_label_vessel_list[instance_id]
        label_logs = []
        instance_patch_num, instance_pos_num = 0, 0
        for label in ['fp', 'low', 'cal', 'mix', 'stent']:
            if len(label_vessel_list[label]) > 0:
                vessel_list = label_vessel_list[label]
                vessel_strs = ['%s=%d' % (vessel, len(vessel_list[vessel])) for vessel in sorted(vessel_list.keys())]
                total_num = sum([len(l) for l in vessel_list.values()])
                instance_patch_num += total_num
                if label != 'fp':
                    instance_pos_num += total_num
                label_logs.append('    label=%s, patch=%d: %s' % (label, total_num, '/'.join(vessel_strs)))
        print('%s, %d patches, %d positive patches' % (instance_id, instance_patch_num, instance_pos_num))
        for log in label_logs:
            print('%s' % log)
        print('')


if __name__ == '__main__':
    patch_main()
    # list_main()
    # parse_list()

