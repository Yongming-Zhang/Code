from cta_util import *
import os
import glob
import numpy
import nibabel as nib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from seg.skeleton import split_skeleton_segments
from lib.utils.mio import load_string_list, dicom_load, json_load, json_save, pickle_load, pickle_save, mkdir_safe
from eval.vessel import eval_vessel


def load_ai_sub_dir_path_dict(sub_dirs):
    cta_dirs = []
    for sub_dir in sub_dirs:
        cta_dirs += glob.glob('/data1/inputdata/%s/*/*_CTA/' % sub_dir.split('_')[0])
    sub_dir_ai_path_dict = {}
    for cta_dir in cta_dirs:
        dcm_paths = glob.glob(cta_dir[:-5] + '/*.dcm')
        if len(dcm_paths) > 0:
            dcm = dicom_load(dcm_paths[0])
            sub_dir = dcm.PatientID + '_' + dcm.StudyInstanceUID + '_' + dcm.SeriesInstanceUID
            sub_dir_ai_path_dict[sub_dir] = cta_dir
    return sub_dir_ai_path_dict


def collect_gt_results(base_dir, sub_dirs):
    gt_results = {}
    match_num, miss_num = 0, 0
    for sub_dir in sub_dirs:
        vm_path = base_dir + 'ann/v0602/' + sub_dir + '.nii.gz'
        if os.path.exists(vm_path):
            gt_results[sub_dir] = {
                'vessel_mask_path': vm_path
            }
            match_num += 1
        else:
            gt_results[sub_dir] = None
            miss_num += 1
    print('loaded %d gt_results, miss %d' % (match_num, miss_num))
    return gt_results


def collect_ai_results(sub_dirs):
    sub_dir_ai_path_dict = load_ai_sub_dir_path_dict(sub_dirs)
    match_num, miss_num = 0, 0
    ai_results = {}
    for sub_dir in sub_dirs:
        if sub_dir in sub_dir_ai_path_dict:
            match_num += 1
            ai_dir = sub_dir_ai_path_dict[sub_dir]
            ai_results[sub_dir] = {
                'vessel_mask_path': ai_dir + 'mask_source/mask_vessel.nii.gz',
                'centerline_dir': 'centerline/'
            }
        else:
            ai_results[sub_dir] = None
            miss_num += 1
    print('loaded %d ai_results, miss %d' % (match_num, miss_num))
    return ai_results


def format_eval_result(result):
    # vessel
    vessel_result = result['vessel']
    dice_all, dice_req, pixel_all_recall, pixel_ignore_recall, pixel_req_recall, pixel_prec = vessel_result[:6]
    sr, sr_req, sp, cr, clcr = vessel_result[-5:]
    s = 'Vessel: DICE_ALL=%.4f, DICE_REQ=%.4f, ' % (dice_all, dice_req)
    s += 'PR[ALL/IGN/REQ]=%.2f%%/%.2f%%/%.2f%%, PP=%.2f%%. ' % (
        pixel_all_recall * 100, pixel_ignore_recall * 100, pixel_req_recall * 100, pixel_prec * 100)
    s += 'SR_ALL=%.4f, SR/SP=%.4f/%.4f, CR=%.4f, CLCR=%.4f. ' % (sr, sr_req, sp, cr, clcr)
    avg_seg_recall, avg_bridge_recall, avg_end_recall = vessel_result[6:9]
    s += 'AVG_R[SEG/MID/END=%.2f%%/%.2f%%/%.2f%%], ' % (
        avg_seg_recall * 100, avg_bridge_recall * 100, avg_end_recall * 100)
    index = 9
    for recall_thresh in [70, 80, 90]:
        seg_recall, bridge_recall, end_recall = vessel_result[index:(index+3)]
        index += 3
        s += 'R%d[SEG/MID/END=%.2f%%/%.2f%%/%.2f%%], ' % (
            recall_thresh, seg_recall * 100, bridge_recall * 100, end_recall * 100)
    return s


def proc_vessel_mask_gt(gt):
    gt[(gt > 0) * (gt != 3)] = 1
    return gt


def load_vessel_gt(gt_dict):
    vessel_mask_gt = proc_vessel_mask_gt(nib.load(gt_dict['vessel_mask_path']).get_data())
    vessel_gt_ske_path = gt_dict['vessel_mask_path'][:-6] + 'pkl'
    if os.path.exists(vessel_gt_ske_path):
        vessel_gt_ske_dict = pickle_load(vessel_gt_ske_path)
    else:
        ske_pts, seg_pts, seg_types, labels = split_skeleton_segments(vessel_mask_gt)
        vessel_gt_ske_dict = {
            'ske_pts': ske_pts,
            'seg_pts': seg_pts,
            'seg_types': seg_types
        }
        pickle_save(vessel_gt_ske_path, vessel_gt_ske_dict)
    vessel_gt_ske_dict['gt'] = vessel_mask_gt
    return vessel_gt_ske_dict


def eval_one(ai_result, gt_dict, is_flip, sub_dir, save_dir):
    vessel_mask_ai = nib.load(ai_result['vessel_mask_path']).get_data()
    if is_flip:
        vessel_mask_ai = vessel_mask_ai[:, :, ::-1]
    vessel_gt_dict = load_vessel_gt(gt_dict)
    vessel_results = eval_vessel(vessel_mask_ai, vessel_gt_dict)
    result = {
        'vessel': vessel_results
    }
    print('%s: ' % sub_dir)
    print('  %s' % format_eval_result(result))
    json_save(save_dir + sub_dir + '.json', result)
    return result


def reduce_eval_result(results):
    vessel_results = numpy.float32([result['vessel'] for result in results]).mean(axis=0).tolist()
    result = {
        'vessel': vessel_results
    }
    return result


def main_old():
    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    data_info_dict = json_load(base_dir + 'config/data_info.json')
    sub_dirs = load_string_list(base_dir + 'config/sub_dirs.txt')
    eval_dir = base_dir + 'eval/0515/'
    mkdir_safe(eval_dir)
    gt_results = collect_gt_results(base_dir, sub_dirs)
    ai_results = collect_ai_results(sub_dirs)
    max_num_workers = 0, 16
    ex = ProcessPoolExecutor(max_workers=8)
    eval_results_dict = {}
    submitted_sub_dirs = []
    for sid, sub_dir in enumerate(sub_dirs):
        ai, gt = ai_results[sub_dir], gt_results[sub_dir]
        if ai is None or gt is None:
            print('Skipped %s, has_gt=%s, has_ai=%s' % (sub_dir, gt is None, ai is None))
            continue
        future = ex.submit(eval_one, ai, gt, data_info_dict[sub_dir]['flip'], sub_dir, eval_dir)
        eval_results_dict[sub_dir] = future
        # eval_results_dict[sub_dir] = eval_one(ai, gt, data_info_dict[sub_dir]['flip'])
        submitted_sub_dirs.append(sub_dir)
        if len(submitted_sub_dirs) == max_num_workers:
            ex.shutdown(wait=True)
            for sub_dir_i in submitted_sub_dirs:
                eval_results_dict[sub_dir_i] = eval_results_dict[sub_dir_i].result()
                print('%s: ' % sub_dir_i)
                print('  %s' % format_eval_result(eval_results_dict[sub_dir_i]))
                json_save(eval_dir + sub_dir_i + '.json', eval_results_dict[sub_dir_i])
            ex = ProcessPoolExecutor()
            submitted_sub_dirs = []
        # print('%s: ' % sub_dir)
        # print('  %s' % format_eval_result(eval_results_dict[sub_dir]))
    for sub_dir_i in submitted_sub_dirs:
        eval_results_dict[sub_dir_i] = eval_results_dict[sub_dir_i].result()
        print('%s: ' % sub_dir_i)
        print('  %s' % format_eval_result(eval_results_dict[sub_dir_i]))
        json_save(eval_dir + sub_dir_i + '.json', eval_results_dict[sub_dir_i])
    eval_result_avg = reduce_eval_result(eval_results_dict.values())
    print('AVG: %s' % format_eval_result(eval_result_avg))


def main():
    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    data_info_dict = json_load(base_dir + 'config/data_info.json')
    sub_dirs = load_string_list(base_dir + 'config/test_sub_dirs.txt')
    eval_dir = base_dir + 'eval/0630_new/'
    mkdir_safe(eval_dir)
    gt_results = collect_gt_results(base_dir, sub_dirs)
    ai_results = collect_ai_results(sub_dirs)
    ex = ProcessPoolExecutor(max_workers=8)
    eval_results_dict = {}
    #for sid, sub_dir in enumerate(sub_dirs):
    #    ai, gt = ai_results[sub_dir], gt_results[sub_dir]
    #    if ai is None or gt is None:
    #        continue
    #    flip = data_info_dict[sub_dir]['flip']
    #    if data_info_dict[sub_dir]['kernel'] == 'STANDARD':
    #        flip = 1 - flip
    #    eval_one(ai_results[sub_dir], gt_results[sub_dir], flip, sub_dir, eval_dir)
    
    for sid, sub_dir in enumerate(sub_dirs):
        ai, gt = ai_results[sub_dir], gt_results[sub_dir]
        if ai is None or gt is None:
            print('Skipped %s, has_gt=%s, has_ai=%s' % (sub_dir, gt is None, ai is None))
            continue
        flip = data_info_dict[sub_dir]['flip']
        if data_info_dict[sub_dir]['kernel'] == 'STANDARD':
            flip = 1 - flip
        future = ex.submit(eval_one, ai, gt, flip, sub_dir, eval_dir)
        eval_results_dict[sub_dir] = future
    ex.shutdown(wait=True)
    eval_results_dict = {k: v.result() for k, v in eval_results_dict.items()}
    eval_result_avg = reduce_eval_result(eval_results_dict.values())
    print('AVG: %s' % format_eval_result(eval_result_avg))


def eval_result_to_dict(result):
    vessel_result = result['vessel']
    dice_all, dice_req, pixel_all_recall, pixel_ignore_recall, pixel_req_recall, pixel_prec = vessel_result[:6]
    sr, sr_req, sp, cr, clcr = vessel_result[-5:]
    avg_seg_recall, avg_bridge_recall, avg_end_recall = vessel_result[6:9]
    d = {
        'DICE_ALL': '%.4f' % dice_all,
        'DICE_REQ': '%.4f' % dice_req,
        'PR_ALL': '%.2f%%' % (pixel_all_recall * 100),
        'PR_IGN': '%.2f%%' % (pixel_ignore_recall * 100),
        'PR_REQ': '%.2f%%' % (pixel_req_recall * 100),
        'PP': '%.2f%%' % (pixel_prec * 100),
        'SR_ALL': '%.2f%%' % (sr * 100),
        'SR_REQ': '%.2f%%' % (sr_req * 100),
        'SP': '%.2f%%' % (sp * 100),
        'CR': '%.4f' % cr,
        'CLCR': '%.4f' % clcr,
        'SEG_R': '%.2f%%' % (avg_seg_recall * 100),
        'MID_R': '%.2f%%' % (avg_bridge_recall * 100),
        'END_R': '%.2f%%' % (avg_end_recall * 100)
    }
    index = 9
    for recall_thresh in [70, 80, 90]:
        seg_recall, bridge_recall, end_recall = vessel_result[index:(index+3)]
        index += 3
        d['SEG_R%d' % recall_thresh] = '%.2f%%' % (seg_recall * 100)
        d['MID_R%d' % recall_thresh] = '%.2f%%' % (bridge_recall * 100)
        d['END_R%d' % recall_thresh] = '%.2f%%' % (end_recall * 100)
    return d


def count():
    base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    data_info_dict = json_load(base_dir + 'config/data_info.json')
    sub_dirs = load_string_list(base_dir + 'config/test_sub_dirs.txt')
    eval_dir = base_dir + 'eval/0515/'
    results_by_attr = defaultdict(list)
    kernels = []
    for sub_dir in sub_dirs:
        if not os.path.exists(eval_dir + sub_dir + '.json'):
            continue
        result = json_load(eval_dir + sub_dir + '.json')
        print('%s: ' % sub_dir)
        print('  %s' % format_eval_result(result))
        data_info = data_info_dict[sub_dir]
        kernel = data_info['kernel']
        if kernel == 'B30f':
            continue
        # if kernel in ['FC03', 'FC43', 'FC05']:
        #     kernel = ''
        man_kernel = data_info['man'] + '/' + kernel
        kernels.append(man_kernel)
        results_by_attr[man_kernel].append(result)
        results_by_attr['TOTAL'].append(result)
    col_names = ['CASE_NUM', 'DICE_ALL', 'DICE_REQ', 'PR_ALL', 'PR_IGN', 'PR_REQ', 'PP', 'SR_ALL', 'SR_REQ', 'SP', 'CR', 'CLCR']
    for recall_thresh in ['', '70', '80', '90']:
        col_names += ['SEG_R' + recall_thresh, 'MID_R' + recall_thresh, 'END_R' + recall_thresh]
    row_names = sorted(list(set(kernels))) + ['TOTAL']
    value_dict = {}
    for row_name in row_names:
        result = reduce_eval_result(results_by_attr[row_name])
        rd = eval_result_to_dict(result)
        rd['CASE_NUM'] = len(results_by_attr[row_name])
        value_dict[row_name] = rd
    lines = gen_table_strings(value_dict, col_name_len=15, col_unit_len=9, align='c', row_names=row_names, col_names=col_names)
    print('Vessel Segmentation Evaluation on STD128 of AI_VERSION=20200515')
    for line in lines:
        print(line)


if __name__ == '__main__':
    #main()
    count()

