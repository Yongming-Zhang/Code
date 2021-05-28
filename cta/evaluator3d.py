import os
import sys
import numpy
from collections import defaultdict
from global_var import plaque_type_dict, segment_vessel_dict
from cta_util import *
from lib.utils.meter import SizeSpliter
from lib.utils.mio import json_load, load_string_list


class PlaqueEvaluator(object):
    def __init__(self, name=''):
        self.eval_name = name
        self.vessel_num = 0
        self.merged_plaque3d_list_dict = {}
        self.gt_ps_docid_dict = {}
        self.ps_cl_dicts = {}
        self.les_types = ['cal', 'low', 'mix']

    def load_gt(self, jd_list, centerline_dir, ps_list=None):
        self.ps_cl_dicts = {ps: json_load(centerline_dir + ps + '.json') for ps in ps_list}
        ps_plaque_list = {}
        les_type_roi_nums = defaultdict(lambda: 0)
        les_type_les_nums = defaultdict(lambda: 0)
        les_num, roi_num = 0, 0
        for jd in jd_list:
            ps = jd['patientID'].split('_')[0] + '_' + jd['studyUID']
            if ps not in self.ps_cl_dicts:
                continue
            if ps not in ps_plaque_list:
                ps_plaque_list[ps] = []
            doc_id = str(jd['other_info']['doctorId'])
            self.gt_ps_docid_dict[ps] = doc_id
            cl_dict = self.ps_cl_dicts[ps]
            for node in jd['nodes']:
                les_type = node['descText'][1][0][0]['select']
                if les_type not in plaque_type_dict:
                    continue
                les_type = plaque_type_dict[les_type]
                for roi in node['rois'] + node['bounds']:
                    les_type_roi_nums[les_type] += 1
                    roi_num += 1
                    size_w, size_h = get_size(roi['edge'])
                    instance_num = str(roi['slice_index'])
                    # print(instance_num, cl_dict['instance_centerline2d_dict'].keys())
                    c2d_pts = numpy.array(cl_dict['instance_centerline2d_dict'][instance_num])
                    vessel_name = cl_dict['instance_vessel_dict'][instance_num]
                    c3d_pts = numpy.array(cl_dict['centerline3d_dict'][vessel_name])
                    pts3d = map_roi_to_3d(roi, c2d_pts, c3d_pts)
                    ps_plaque_list[ps].append(Plaque3D(pts3d, les_type, (size_w + size_h) * 0.5))
        self.merged_plaque3d_list_dict = {ps: Plaque3D.merge(plaque_list) for ps, plaque_list in ps_plaque_list.items()}
        for ps, merged_plaque_list in self.merged_plaque3d_list_dict.items():
            for p in merged_plaque_list:
                les_type_les_nums[p.les_type] += 1
                les_num += 1
        self.vessel_num = len(jd_list)
        case_num = len(ps_plaque_list)
        les_num_str = ', '.join(['%s=%d' % (les_type, num) for les_type, num in les_type_les_nums.items()])
        roi_num_str = ', '.join(['%s=%d' % (les_type, num) for les_type, num in les_type_roi_nums.items()])
        print('loaded %d case, %d vessels, including %d (%s) lesions, %d (%s) rois' % (
            case_num, self.vessel_num, les_num, les_num_str, roi_num, roi_num_str))

    def evaluate(self, predict_dir, score_thresh, iomin_thresh=0.1):
        size_spliter = SizeSpliter([0, 4, 8, 16, 32, 64])
        les_type_dict = {'noncalcified': 'low', 'calcified': 'cal', 'mix': 'mix'}
        ps_list = sorted(self.merged_plaque3d_list_dict.keys())
        fp_nums = defaultdict(lambda: 0)
        pred_nums = defaultdict(lambda: 0)
        find_nums = defaultdict(lambda: defaultdict(lambda: 0))
        total_nums = defaultdict(lambda: defaultdict(lambda: 0))
        case_num = 0
        for ps in ps_list:
            ai_json_path = predict_dir + ps + '.json'
            if not os.path.exists(ai_json_path):
                continue
            case_num += 1
            predict_json = json_load(ai_json_path)
            cl_dict = self.ps_cl_dicts[ps]
            gt_plaque3d_list = self.merged_plaque3d_list_dict[ps]
            gt_find_mask = [0] * len(gt_plaque3d_list)
            cur_fp_num, cur_pred_num = 0, 0
            for plaque_dict in predict_json['plaque']:
                score = plaque_dict['score2'] if sys.argv[3] == '2' else plaque_dict['score']
                if score < score_thresh:
                    continue
                vessel_name = plaque_dict['position']
                segment = plaque_dict['segment']
                if segment_vessel_dict[segment] != vessel_name:
                    continue
                if plaque_dict['plaque_type'] not in les_type_dict:
                    continue
                pred_les_type = les_type_dict[plaque_dict['plaque_type']]
                pred_nums[pred_les_type] += 1
                cur_pred_num += 1
                c3d_pts = cl_dict['centerline3d_dict'][vessel_name]
                pred_pts3d = get_rounded_pts(c3d_pts, plaque_dict['merged_centerline_range'], as_unique=True)
                is_fp = True
                for gt_id, gt_p3d in enumerate(gt_plaque3d_list):
                    if cl_pts_iou(gt_p3d.pts3d, pred_pts3d) >= iomin_thresh:
                        is_fp = False
                        gt_find_mask[gt_id] = 1
                if is_fp:
                    cur_fp_num += 1
                    fp_nums[pred_les_type] += 1
            recall_strs = []
            for is_find, gt_p3d in zip(gt_find_mask, gt_plaque3d_list):
                size_str = size_spliter.get_size_str(gt_p3d.size)
                gt_type = gt_p3d.les_type
                for row in [gt_type, 'total']:
                    for col in [size_str, 'ALL']:
                        total_nums[row][col] += 1
                        if is_find:
                            find_nums[row][col] += 1
                recall_strs.append('%s/%.1f=%s' % (gt_type, gt_p3d.size, 'find' if is_find else 'miss'))

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
        for k in ['cal', 'low', 'mix']:
            fp_detail_list.append('%s=%d(%.2f%%)' % (k, fp_nums[k], fp_nums[k] * 100. / total_fp_num))
        fp_str += '[%s]' % (', '.join(fp_detail_list))
        print('%s. %s. %s' % (summary_str, metric_str, fp_str))
        # recall strings
        recall_str_table = defaultdict(lambda: defaultdict(lambda: '0'))
        row_names = self.les_types + ['total']
        col_names = ['ALL'] + size_spliter.get_all_size_strings()
        for row_name in row_names:
            for col_name in col_names:
                recall_str_table[row_name][col_name] = fmt_ratio_str(
                    find_nums[row_name][col_name], total_nums[row_name][col_name])
        recall_lines = gen_table_strings(
            recall_str_table, '', align='c', col_unit_len=19, row_names=row_names, col_names=col_names)
        for line in recall_lines:
            print('%s' % line)


def test():
    pe = PlaqueEvaluator()
    centerline_dir = '/breast_data/cta/centerline_dicts/'
    task_names = ['task_10626_0', 'task_10652_0-1582016734', 'task_10627_0-1586145902']
    # task_names = ['task_10626_0']
    jd_list = []
    for task_name in task_names:
        jd_list += json_load('/breast_data/cta/annotation/%s.json' % task_name)
    ps_list = load_string_list('/breast_data/cta/new_data/b1_bt_587/config/valid_psid_list.txt')
    pe.load_gt(jd_list, centerline_dir, ps_list)
    predict_dir = '/breast_data/cta/new_data/b1_bt_587/ai_results/0422_v2_t%s/json/' % sys.argv[1]
    score_thresh = float(sys.argv[2])
    pe.evaluate(predict_dir, score_thresh)


if __name__ == '__main__':
    test()

