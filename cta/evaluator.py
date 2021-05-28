from collections import defaultdict
from lib.utils.rect import rect_iou
from global_var import plaque_type_dict
from cta_util import get_bbox, get_size, get_pred_bbox_list, fmt_ratio_str, gen_table_strings
from lib.utils.meter import SizeSpliter


class PlaqueEvaluator(object):
    def __init__(self, name=''):
        self.eval_name = name
        self.gt_roi_type_dict = defaultdict(list)
        self.gt_les_type_dict = defaultdict(list)
        self.gt_pssi_roi_dict = defaultdict(list)
        self.gt_pss_docid_dict = {}
        self.les_types = ['cal', 'low', 'mix']

    def load_gt(self, jd_list):
        self.gt_roi_type_dict = defaultdict(list)
        self.gt_les_type_dict = defaultdict(list)
        self.gt_pssi_roi_dict = defaultdict(list)
        les_num, les_roi_num = 0, 0
        for jd in jd_list:
            pss = jd['patientID'] + '_' + jd['studyUID'] + '_' + jd['seriesUID']
            doc_id = str(jd['other_info']['doctorId'])
            self.gt_pss_docid_dict[pss] = doc_id
            for node in jd['nodes']:
                les_type = node['descText'][1][0][0]['select']
                if les_type not in plaque_type_dict:
                    continue
                les_type = plaque_type_dict[les_type]
                les_index = len(self.gt_les_type_dict[les_type])
                les_info = {'index': les_index, 'roi_indices': []}
                les_num += 1
                for roi in node['rois'] + node['bounds']:
                    les_roi_num += 1
                    roi_bbox = get_bbox(roi['edge'])
                    size = get_size(roi['edge'])
                    pssi = pss + '_%d' % roi['slice_index']
                    pssi_index = len(self.gt_pssi_roi_dict[pssi])
                    roi_index = len(self.gt_roi_type_dict[les_type])
                    les_info['roi_indices'].append(roi_index)
                    self.gt_pssi_roi_dict[pssi].append({
                        'bbox': roi_bbox,
                        'size': [min(size), max(size)],
                        'type': les_type,
                        'doc_id': doc_id,
                        'les_index': les_index,
                        'roi_index': roi_index
                    })
                    self.gt_roi_type_dict[les_type].append({
                        'bbox': roi_bbox,
                        'size': [min(size), max(size)],
                        'les_index': les_index,
                        'doc_id': doc_id,
                        'pssi': pssi,
                        'pssi_index': pssi_index
                    })
                self.gt_les_type_dict[les_type].append(les_info)
        les_num_str = ', '.join(['%s=%d' % (les_type, len(lesions)) for les_type, lesions in self.gt_les_type_dict.items()])
        roi_num_str = ', '.join(['%s=%d' % (les_type, len(rois)) for les_type, rois in self.gt_roi_type_dict.items()])
        print('loaded %d jsons, including %d (%s) lesions, %d (%s) rois' % (
            len(jd_list), les_num, les_num_str, les_roi_num, roi_num_str))

    def evaluate(self, pssi_pred_dict, score_thresh, criteria='iou', iou_thresh=0.1, is_combine=False, is_verbose=False):
        case_num = len(set(['/'.join(pssi.split('_')[:-2]) for pssi in pssi_pred_dict.keys()]))
        pssi_eval_info, find_roi_flags, find_les_flags, total_fp_num, total_pred_num, image_num, les_type_fp_nums = \
            self._collect_eval_flags(pssi_pred_dict, score_thresh, criteria, iou_thresh, is_combine, is_verbose)
        self._print_summary_metric_fp(case_num, image_num, criteria, iou_thresh, score_thresh, is_combine,
                                      total_fp_num, total_pred_num, les_type_fp_nums)
        seg_recall_lines = self._eval_recall_by_seg(find_les_flags)
        roi_recall_lines = self._eval_recall_by_roi(find_roi_flags)
        for sl, rl in zip(seg_recall_lines, roi_recall_lines):
            print('  %s|%s' % (sl, '|'.join(rl.split('|')[2:])))
        return pssi_eval_info

    def _collect_eval_flags(self, pssi_pred_dict, score_thresh, criteria, iou_thresh, is_combine, is_verbose):
        label_dict = {1: 'low', 2: 'cal', 3: 'mix'}
        find_roi_flags = defaultdict(lambda: defaultdict(lambda: 0))
        find_les_flags = defaultdict(lambda: defaultdict(lambda: 0))
        total_fp_num, total_pred_num, image_num = 0, 0, len(pssi_pred_dict)
        les_type_fp_nums = defaultdict(lambda: 0)
        pssi_eval_info = defaultdict(lambda: dict)
        for pssi in sorted(pssi_pred_dict.keys()):
            pred_bbox_list = get_pred_bbox_list(pssi_pred_dict[pssi], score_thresh, is_combine)
            pssi_pred_num = len(pred_bbox_list)
            pred_correct_flags = [0] * pssi_pred_num
            pssi_find_indices, pssi_find_num, pssi_gt_num = [], 0, len(self.gt_pssi_roi_dict[pssi])
            for gt_id, gt_roi_dict in enumerate(self.gt_pssi_roi_dict[pssi]):
                gt_les_type = gt_roi_dict['type']
                roi_index, les_index = gt_roi_dict['roi_index'], gt_roi_dict['les_index']
                is_find = False
                for pred_id, pred_bbox in enumerate(pred_bbox_list):
                    # iou = rect_iou(gt_roi_dict['bbox'], pred_bbox)
                    iou = rect_iou(gt_roi_dict['bbox'], pred_bbox['bbox'], criteria)
                    if iou >= iou_thresh:
                        pred_correct_flags[pred_id] = 1
                        is_find = True
                if is_find:
                    find_roi_flags[gt_les_type][roi_index] = 1
                    find_les_flags[gt_les_type][les_index] = 1
                    pssi_find_indices.append(gt_id)
                    pssi_find_num += 1
                else:
                    min_size, max_size = gt_roi_dict['size']
                    if min_size > 8 or max_size > 16:
                        print('\t\t%s: miss %dx%d' % (pssi, min_size, max_size))
            for pred_id, pred_bbox in enumerate(pred_bbox_list):
                if pred_correct_flags[pred_id] == 0:
                    les_type_fp_nums[label_dict[pred_bbox['category_id']]] += 1
            pssi_fp_num = pssi_pred_num - sum(pred_correct_flags)
            total_fp_num += pssi_fp_num
            total_pred_num += pssi_pred_num
            pss = '_'.join(pssi.split('_')[:-1])
            pssi_eval_info[pssi] = {
                'find_indices': pssi_find_indices,
                'gt_info': {'doc_id': self.gt_pss_docid_dict[pss], 'roi_dicts': self.gt_pssi_roi_dict[pssi]},
                'pred_bbox_list': pred_bbox_list,
                'pred_correct_flags': pred_correct_flags
            }
            if is_verbose:
                print('\t%s: find/gt=%d/%d, fp/pred=%d/%d' % (
                    pssi, pssi_find_num, pssi_gt_num, pssi_fp_num, pssi_pred_num))
        return pssi_eval_info, find_roi_flags, find_les_flags, total_fp_num, total_pred_num, image_num, les_type_fp_nums

    def _print_summary_metric_fp(self, case_num, image_num, criteria, iou_thresh, score_thresh, is_combine,
                                 total_fp_num, total_pred_num, les_type_fp_nums):
        combine_str = ', MERGE_BY_IOMIN>0.5' if is_combine else ''
        summary_str = 'Summary for %s: %d cases' % (self.eval_name, case_num)
        metric_str = '%s=%s, score_thresh=%s%s' % (criteria.upper(), str(iou_thresh), str(score_thresh), combine_str)
        pred_prec_str = fmt_ratio_str(total_pred_num - total_fp_num, total_pred_num)
        fp_per_img = total_fp_num * 1. / image_num
        fp_str = 'Prec=%s, FP=%.2f/image(%d/%d)' % (pred_prec_str, fp_per_img, total_fp_num, image_num)
        fp_detail_list = []
        for k in ['cal', 'low', 'mix']:
            fp_detail_list.append('%s=%d(%.2f%%)' % (k, les_type_fp_nums[k], les_type_fp_nums[k] * 100. / total_fp_num))
        fp_str += '[%s]' % (', '.join(fp_detail_list))
        print('%s. %s. %s' % (summary_str, metric_str, fp_str))

    def _eval_recall_by_seg(self, find_les_flags):
        total_find_num, total_gt_num = 0, 0
        recall_str_table = defaultdict(lambda: defaultdict(lambda: '0'))
        for les_type in self.les_types:
            find_num = sum(find_les_flags[les_type].values())
            gt_num = len(self.gt_les_type_dict[les_type])
            recall_str_table[les_type]['R@SEG'] = fmt_ratio_str(find_num, gt_num)
            total_find_num += find_num
            total_gt_num += gt_num
        recall_str_table['total']['R@SEG'] = fmt_ratio_str(total_find_num, total_gt_num)
        row_names = self.les_types + ['total']
        lines = gen_table_strings(
            recall_str_table, '', align='c', col_unit_len=19, row_names=row_names, col_names=['R@SEG'])
        return lines

    def _eval_recall_by_roi(self, find_roi_flags):
        size_spliter = SizeSpliter([0, 4, 8, 16, 32, 64])
        size_strings = ['ROI/' + s for s in size_spliter.get_all_size_strings()]
        find_nums = defaultdict(lambda: defaultdict(lambda: 0))
        total_nums = defaultdict(lambda: defaultdict(lambda: 0))
        for les_type in self.les_types:
            for roi_id, roi_dict in enumerate(self.gt_roi_type_dict[les_type]):
                is_find = find_roi_flags[les_type][roi_id]
                min_size, max_size = roi_dict['size']
                size_str = 'ROI/' + size_spliter.get_size_str(min_size * 0.5 + max_size * 0.5)
                for row_name in [les_type, 'total']:
                    for col_name in [size_str, 'ROI/ALL']:
                        total_nums[row_name][col_name] += 1
                        if is_find:
                            find_nums[row_name][col_name] += 1
        recall_str_table = defaultdict(lambda: defaultdict(lambda: '0'))
        row_names = self.les_types + ['total']
        col_names = ['ROI/ALL'] + size_strings
        for row_name in row_names:
            for col_name in col_names:
                recall_str_table[row_name][col_name] = fmt_ratio_str(
                    find_nums[row_name][col_name], total_nums[row_name][col_name])
        lines = gen_table_strings(
            recall_str_table, '', align='c', col_unit_len=19, row_names=row_names, col_names=col_names)
        return lines

