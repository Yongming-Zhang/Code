from cta_util import *
import os
import glob
from collections import defaultdict
from lib.utils.meter import SizeSpliter
from lib.utils.mio import load_string_list, json_load, json_save, mkdir_safe


def merge_gt():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    psid_list = load_string_list(base_dir + 'config/test_psid_list.txt')
    raw_ann_dir = base_dir + 'annotation/cta_cpr/raw_json/'
    # raw_ann_dir = '/breast_data/cta/json/train02/'
    dst_ann_dir = base_dir + 'annotation/cta_cpr/merged_p3d_v2/'
    mkdir_safe(dst_ann_dir)
    centerline_dir = base_dir + 'centerline_dicts/'
    json_format = 'zfd'
    valid_les_types = {
        'low': 'low', 'cal': 'cal', 'mix': 'mix', 'clot': 'block',
        'low_density': 'low', 'calc': 'cal',
        'w_stent': 'stent', 'b_stent': 'stent', 'c_stent': 'stent', 'mb_mca': 'xjq'}
    for psid in psid_list:
        json_paths = glob.glob(raw_ann_dir + psid + '_*.json')
        if len(json_paths) == 0:
            continue
        if os.path.exists(dst_ann_dir + psid + '.json'):
            continue
        cl_dict = json_load(centerline_dir + psid + '.json')
        raw_p3d_list, roi_num = [], 0
        for json_path in json_paths:
            jd = json_load(json_path)
            vessel_name = json_path.split('/')[-1].split('_')[-1].split('.')[0]
            # print(json_path, [node['type'] for node in jd['nodes']])
            for node in jd['nodes']:
                les_type = node['type']
                if les_type not in valid_les_types:
                    continue
                les_type = valid_les_types[les_type]
                stenosis = node.get('stenosis', 'none')
                for roi in node['rois']:
                    roi_num += 1
                    if json_format == 'zqy':
                        roi = {'edge': roi, 'slice_index': node['slice_index']}
                    edge, instance_num = roi['edge'], str(roi['slice_index'])
                    size_w, size_h = get_size(edge)
                    c2d_pts = numpy.array(cl_dict['instance_centerline2d_dict'][vessel_name][instance_num])
                    c3d_pts = numpy.array(cl_dict['centerline3d_dict'][vessel_name])
                    pts3d = map_roi_to_3d(roi, c2d_pts, c3d_pts)
                    if pts3d is None:
                        continue
                    raw_p3d_list.append(Plaque3D(pts3d, les_type, (size_w + size_h) * 0.5, 1, stenosis))
        merged_p3d_list = Plaque3D.merge(raw_p3d_list)
        p3d_types = [p3d.les_type for p3d in merged_p3d_list]
        # for p3d in merged_p3d_list:
        #     print('%s:, %d rois, %s' % (p3d.les_type, p3d.roi_num, p3d.pts3d.tolist()))
        merged_p3d_list = [p3d.to_json() for p3d in merged_p3d_list]
        json_save(dst_ann_dir + psid + '.json', merged_p3d_list)
        print('processed %s: %d rois, %d merged p3d, types: %s' % (psid, roi_num, len(merged_p3d_list), p3d_types))


def gen_cta_p3d():
    pass


def prediction_to_p3d_list(base_dir, pred_by_mrcnn, score_thresh):
    centerline_dir = base_dir + 'centerline_dicts/'
    ps_list = list(set('_'.join(psvi.split('_')[:2]) for psvi in pred_by_mrcnn.keys()))
    cl_dicts = {ps: json_load(centerline_dir + ps + '.json') for ps in ps_list}
    ps_plaque_list = defaultdict(list)
    label_dict = {1: 'low', 2: 'cal', 3: 'mix', 4: 'stent', 5: 'xjq'}
    for psvi, pred_list in pred_by_mrcnn.items():
        pid, study, vessel_name, instance_num = psvi.split('_')
        ps = pid + '_' + study
        if ps not in ps_plaque_list:
            ps_plaque_list[ps] = []
        cl_dict = cl_dicts[ps]
        c2d_pts = numpy.array(cl_dict['instance_centerline2d_dict'][vessel_name][instance_num])
        c3d_pts = numpy.array(cl_dict['centerline3d_dict'][vessel_name])
        for pred in pred_list:
            # if pred['category_id'] > 3:
            #     continue
            if pred['score'] >= score_thresh:
                les_type = label_dict[pred['category_id']]
                bbox = pred['bbox']
                pts3d = map_rect_to_3d(bbox, c2d_pts, c3d_pts)
                if pts3d is None:
                    continue
                ps_plaque_list[ps].append(Plaque3D(pts3d, les_type, (bbox[2] + bbox[3]) * 0.5, score=pred['score']))
    print('Merging prediction: %d cases' % len(ps_plaque_list))
    merged_plaque3d_list_dict = {ps: Plaque3D.merge(plaque_list) for ps, plaque_list in ps_plaque_list.items()}
    return merged_plaque3d_list_dict


class PlaqueEvaluator(object):
    def __init__(self, name=''):
        self.eval_name = name
        self.merged_plaque3d_list_dict = {}
        self.les_types = ['cal', 'low', 'mix', 'stent', 'xjq']

    def load_gt(self, p3d_dir, ps_list=None):
        les_type_les_nums = defaultdict(lambda: 0)
        les_num, case_num = 0, 0
        skip_ps = [
            '0784360_1.2.392.200036.9116.2.5.1.37.2417525831.1552522058.747563',
            '0784378_1.2.392.200036.9116.2.5.1.37.2417525831.1571895522.162096',
            '0784343_1.2.392.200036.9116.2.5.1.37.2417525831.1531287041.670800',
            '1036598_1.2.840.113564.118796721496052.58444.636244063436974981.1224',
            '1036590_1.2.840.113564.118796721496052.74064.635827480003962835.421'
        ]
        skip_ps = []
        for ps in ps_list:
            ps_key = '_'.join(ps.split('_')[:2])
            if ps_key in skip_ps:
                continue
            if os.path.exists(p3d_dir + ps + '.json'):
                case_num += 1
                self.merged_plaque3d_list_dict[ps_key] = []
                for p3d_json in json_load(p3d_dir + ps + '.json'):
                    p3d = Plaque3D()
                    p3d.from_json(p3d_json)
                    self.merged_plaque3d_list_dict[ps_key].append(p3d)
                    les_type_les_nums[p3d.les_type] += 1
                    les_num += 1
        les_num_str = ', '.join(['%s=%d' % (les_type, num) for les_type, num in les_type_les_nums.items()])
        print('loaded %d case, including %d (%s) lesions' % (case_num, les_num, les_num_str))

    def evaluate(self, ps_pred_p3d_list_dict, score_thresh, sum_score_thresh, iomin_thresh):
        size_spliter = SizeSpliter([0, 4, 8, 16, 32])
        ps_list = sorted(self.merged_plaque3d_list_dict.keys())
        fp_nums = defaultdict(lambda: 0)
        pred_nums = defaultdict(lambda: 0)
        find_nums = defaultdict(lambda: defaultdict(lambda: 0))
        total_nums = defaultdict(lambda: defaultdict(lambda: 0))
        case_num = 0
        for ps in ps_list:
            if ps not in ps_pred_p3d_list_dict:
                continue
            case_num += 1
            gt_plaque3d_list = self.merged_plaque3d_list_dict[ps]
            gt_find_mask = [0] * len(gt_plaque3d_list)
            cur_fp_num, cur_pred_num = 0, 0
            pred_p3d_list = ps_pred_p3d_list_dict[ps]
            for pred_p3d in pred_p3d_list:
                if pred_p3d.score < sum_score_thresh:
                    continue
                pred_les_type = pred_p3d.les_type
                pred_nums[pred_les_type] += 1
                cur_pred_num += 1
                is_fp = True
                for gt_id, gt_p3d in enumerate(gt_plaque3d_list):
                    # print(gt_p3d.pts3d.shape, pred_p3d.pts3d.shape)
                    if cl_pts_iou(gt_p3d.pts3d, pred_p3d.pts3d, 'min') >= iomin_thresh:
                        is_fp = False
                        gt_find_mask[gt_id] = 1
                if is_fp:
                    cur_fp_num += 1
                    fp_nums[pred_les_type] += 1
                # print(pred_p3d.les_type, pred_p3d.score, numpy.int32(pred_p3d.pts3d).tolist())
            recall_strs = []
            # print(['%.2f' % pred_p3d.score for pred_p3d in pred_p3d_list])
            for is_find, gt_p3d in zip(gt_find_mask, gt_plaque3d_list):
                size_str = size_spliter.get_size_str(gt_p3d.size)
                gt_type = gt_p3d.les_type
                for row in [gt_type, 'total']:
                    for col in [size_str, gt_p3d.stenosis, 'ALL']:
                        total_nums[row][col] += 1
                        if is_find:
                            find_nums[row][col] += 1
                recall_strs.append('%s/%s/%.1f=%s' % (gt_type, gt_p3d.stenosis, gt_p3d.size, 'find' if is_find else 'miss'))
            if True:
                print('\t%s: find=%d/%d[%s], fp=%d/%d' % (
                    ps, sum(gt_find_mask), len(gt_find_mask), ', '.join(recall_strs), cur_fp_num, cur_pred_num))
        # summary and fp
        summary_str = 'Summary for %s: %d cases' % (self.eval_name, case_num)
        metric_str = 'IOMIN=%s, score_thresh=%s' % (str(iomin_thresh), str(score_thresh))
        total_pred_num = sum(pred_nums.values())
        total_fp_num = sum(fp_nums.values())
        fp_rate_str = fmt_ratio_str(total_fp_num, total_pred_num)
        fp_per_case = total_fp_num * 1. / case_num
        fp_str = 'FP_rate=%s, FP=%.2f/case(%d/%d)' % (fp_rate_str, fp_per_case, total_fp_num, case_num)
        fp_detail_list = []
        for k in self.les_types:
            fp_detail_list.append('%s=%d(%.2f%%)' % (k, fp_nums[k], fp_nums[k] * 100. / total_fp_num))
        fp_str += '[%s]' % (', '.join(fp_detail_list))
        print('%s. %s. %s' % (summary_str, metric_str, fp_str))
        # recall strings
        recall_str_table = defaultdict(lambda: defaultdict(lambda: '0'))
        row_names = self.les_types + ['total']
        col_names = ['ALL'] + g_stenosis_type_list + size_spliter.get_all_size_strings()
        for row_name in row_names:
            for col_name in col_names:
                recall_str_table[row_name][col_name] = fmt_ratio_str(
                    find_nums[row_name][col_name], total_nums[row_name][col_name])
        recall_lines = gen_table_strings(
            recall_str_table, '', align='c', col_unit_len=17, row_names=row_names, col_names=col_names)
        for line in recall_lines:
            print('%s' % line)


def save_prediction(save_path, ps_pred_p3d_list_dict):
    d = {ps: [p.to_json() for p in v] for ps, v in ps_pred_p3d_list_dict.items()}
    json_save(save_path, d)


def load_p3d_from_json(jd):
    p3d = Plaque3D()
    p3d.from_json(jd)
    return p3d


def load_prediction(path):
    return {ps: [load_p3d_from_json(p) for p in v] for ps, v in json_load(path).items()}


def get_prediction(base_dir, set_names, model_dir, dataset, score_thresh):
    psid_list, pred_by_mrcnn_ = [], {}
    for set_name in set_names:
        pred_path = model_dir + 'cta_train0%s_%s.json' % (dataset[1:], set_name)
        pred_by_mrcnn_.update(json_load(pred_path))
        psid_list += load_string_list(base_dir + 'config/%s_psid_list.txt' % set_name)
    pred_save_dir = base_dir + 'ai_results/temp/plaque_detect_pred/'
    save_path = pred_save_dir + model_dir.replace('/', '_') + '_'.join(set_names) + '_%s.json' % score_thresh
    if os.path.exists(save_path):
        ps_pred_p3d_list_dict = load_prediction(save_path)
    else:
        ps_pred_p3d_list_dict = prediction_to_p3d_list(base_dir, pred_by_mrcnn_, float(score_thresh))
        mkdir_safe(pred_save_dir)
        save_prediction(save_path, ps_pred_p3d_list_dict)
    return psid_list, ps_pred_p3d_list_dict


def test_one_dataset(dataset_name, model_dir, set_names, pe, score_thresh):
    dataset_name_dict = {
        'b2': 'b2_sg_407',
        'b3': 'b3_azcto_150'
    }
    base_dir = '/breast_data/cta/new_data/%s/' % dataset_name_dict[dataset_name]
    ann_dir = base_dir + 'annotation/cta/p3d_s3_checked_json/'
    psid_list, ps_pred_p3d_list_dict = get_prediction(base_dir, set_names, model_dir, dataset_name, score_thresh)
    pe.load_gt(ann_dir, psid_list)
    return psid_list, ps_pred_p3d_list_dict


def test():
    base_dir = '/breast_data/cta/new_data/b2_sg_407/'
    # model_dir = '/breast_data/zhangqianyi/models/cta_200516_mask/inference/model_0080000/'
    # model_dir = '/breast_data/zhangqianyi/models/cta/cta_200418_mask_cas/inference/model_0050000/'
    # model_dir = '/breast_data/zhangqianyi/models/cta_200524_mask/inference/model_0080000/'
    # model_dir = '/breast_data/zhangqianyi/models/cta_200526_mask/inference/model_0065000/'
    # model_dir = '/breast_data/zhangqianyi/models/cta_200608_cluster_mask_cas/inference/model_0040000/'
    # model_dir = '/breast_data/cta/new_data/b2_sg_407/model_data/stage1/cta_200608_cluster_mask_cas/'
    # model_dir = '/breast_data/gaofei/data/ccta/preds/'
    # pred_path = model_dir + 'cta_train02_valid/cta_train02_valid.json'
    score_thresh = sys.argv[2]
    sum_score_thresh = 0 if len(sys.argv) <= 2 else float(sys.argv[3])
    iomin_thresh = float(sys.argv[4])
    model_dir = '/breast_data/gaofei/data/ccta/preds/0912/'
    set_names = ['valid', 'test']
    psid_list, ps_pred_p3d_list_dict = get_prediction(base_dir, set_names, model_dir, 'b2', score_thresh)
    # ann_dir = base_dir + 'annotation/cta_cpr/merged_p3d_v2/'
    ann_dir = base_dir + 'annotation/cta/p3d_s3_json/'
    pe = PlaqueEvaluator()
    pe.load_gt(ann_dir, psid_list)
    pe.evaluate(ps_pred_p3d_list_dict, score_thresh, sum_score_thresh, iomin_thresh)


def test_main():
    pe = PlaqueEvaluator()
    model_dir = '/breast_data/gaofei/data/ccta/preds/1101/'
    # model_dir = '/breast_data/gaofei/data/ccta/preds/0912/'
    set_names = ['valid', 'test']
    dataset_names = sys.argv[1].split('+')
    score_thresh = sys.argv[2]
    sum_score_thresh = 0 if len(sys.argv) <= 2 else float(sys.argv[3])
    iomin_thresh = float(sys.argv[4])
    all_psid_list, all_ps_pred_p3d_list_dict = [], {}
    for dataset_name in dataset_names:
        psid_list, ps_pred_p3d_list_dict = test_one_dataset(dataset_name, model_dir, set_names, pe, score_thresh)
        all_psid_list += psid_list
        all_ps_pred_p3d_list_dict.update(ps_pred_p3d_list_dict)
    pe.evaluate(all_ps_pred_p3d_list_dict, score_thresh, sum_score_thresh, iomin_thresh)


if __name__ == '__main__':
    # merge_gt()
    test_main()

