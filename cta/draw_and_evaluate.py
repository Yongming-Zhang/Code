from cta_util import *
import json
import cv2
import glob
import numpy
import SimpleITK as sitk
from evaluator import PlaqueEvaluator
from lib.utils.mio import *
from lib.image.image import gray2rgb
from lib.image.draw import draw_texts


les_type_color_dict = {
    'cal': (0, 255, 0),
    'low': (0, 0, 255),
    'mix': (0, 97, 255),
}
DRAW_BBOX_THICKNESS = 2
DRAW_BBOX_TEXT_FONT_SCALE = 0.5
DRAW_TITLE_FONT_SCALE = 0.7
PAD_X, PAD_Y = 0, int(DRAW_TITLE_FONT_SCALE * 40)


def load_gt_jsons():
    json_dir = '/breast_data/cta/annotation/'
    task_names = ['task_10626_0', 'task_10652_0-1582016734']
    # task_names = ['task_10626_0']
    jd_list = []
    for task_name in task_names:
        cur_jd_list = json.load(open(json_dir + '%s.json' % task_name))
        if task_name == 'task_10626_0':
            jd_list += cur_jd_list
        elif 'task_10652_0' in task_name:
            for jd in cur_jd_list:
                if '0657701' <= jd['patientID'] <= '0657726':
                    continue
                jd_list.append(jd)
    return jd_list


def gen_images():
    base_dir = '/breast_data/cta/'
    dcm_dir = base_dir + 'dicom/'
    # dataset_names = ['train01', 'test00']
    dataset_names = ['train00']
    min_max_values = [(-100, 700), (0, 600), (400, 1000), ('min', 'max')]
    for dataset_name in dataset_names:
        dcm_paths = sorted(glob.glob(dcm_dir + dataset_name + '/*/*/*.dcm'))
        print('%s: %d dcm' % (dataset_name, len(dcm_paths)))
        for min_value, max_value in min_max_values:
            save_dir = base_dir + 'image/%s_%s/%s/' % (min_value, max_value, dataset_name)
            mkdir_safe(save_dir)
        for did, dcm_path in enumerate(dcm_paths):
            raw_img16 = numpy.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(dcm_path)))
            dcm = dicom_load(dcm_path)
            pid, series, name = dcm_path.split('/')[-3:]
            img_name = dcm.PatientID + '_' + dcm.StudyInstanceUID + '_' + series + '_' + str(dcm.InstanceNumber) + '.png'
            for min_value, max_value in min_max_values:
                img16 = raw_img16.copy()
                img_path = base_dir + 'image/%s_%s/%s/%s' % (min_value, max_value, dataset_name, img_name)
                min_value = img16.min() if min_value == 'min' else min_value
                max_value = img16.max() if max_value == 'max' else max_value
                img16[img16 > max_value] = max_value
                img16[img16 < min_value] = min_value
                img8u = numpy.uint8((img16 - min_value) * 255. / (max_value - min_value))
                save_image(img_path, img8u)
                print('No.%d/%d %s~%s %s' % (did, len(dcm_paths), min_value, max_value, img_path))


def load_pssi_dataset_name():
    dataset_names = ['train01', 'test00']
    pssi_dataset_dict = {}
    for dataset_name in dataset_names:
        img_paths = glob.glob('/breast_data/cta/image/min_max/%s/*.png' % dataset_name)
        for img_path in img_paths:
            pssi_dataset_dict[img_path.split('/')[-1][:-4]] = dataset_name
    return pssi_dataset_dict


def draw_title(img_rgb, title):
    draw_texts(img_rgb, (0, 0), [title], (255, 255, 0), is_bold=True, direct=-1, font_scale=DRAW_TITLE_FONT_SCALE)


def draw_gt_image(img_rgb, gt_info):
    les_nums = {'cal': 0, 'low': 0, 'mix': 0}
    for gid, gt_roi_dict in enumerate(gt_info['roi_dicts']):
        x, y, w, h = gt_roi_dict['bbox']
        les_type = gt_roi_dict['type']
        les_nums[les_type] += 1
        color = les_type_color_dict[les_type]
        x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        cv2.rectangle(img_rgb, (x0 + PAD_X, y0 + PAD_Y), (x1 + PAD_X, y1 + PAD_Y), color=color, thickness=DRAW_BBOX_THICKNESS)
        text = 'No.%d, %s, %dx%d' % (gid, les_type, int(round(w)), int(round(h)))
        draw_texts(img_rgb, (x0 + PAD_X, y0 + PAD_Y), [text], color, font_scale=DRAW_BBOX_TEXT_FONT_SCALE)
    title = 'Doc=%s, cal=%d, low=%d, mix=%d' % (gt_info['doc_id'], les_nums['cal'], les_nums['low'], les_nums['mix'])
    draw_title(img_rgb, title)
    return img_rgb


def draw_pred_list(img_rgb, pred_list):
    label_dict = {1: 'low', 2: 'cal', 3: 'mix'}
    for pred_id, pred_dict in enumerate(pred_list):
        x, y, w, h = pred_dict['bbox']
        x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        pred_type = label_dict[pred_dict['category_id']]
        color = les_type_color_dict[pred_type]
        cv2.rectangle(img_rgb, (x0 + PAD_X, y0 + PAD_Y), (x1 + PAD_X, y1 + PAD_Y), color=color, thickness=DRAW_BBOX_THICKNESS)
        text = 'No.%d, s=%.4f, %s' % (pred_id, pred_dict['score'], pred_type)
        draw_texts(img_rgb, (x0 + PAD_X, y0 + PAD_Y), [text], color, font_scale=DRAW_BBOX_TEXT_FONT_SCALE)
    return img_rgb


def draw_pred_image(img_rgb, eval_info, name):
    les_gt_nums = {'cal': 0, 'low': 0, 'mix': 0}
    les_find_nums = {'cal': 0, 'low': 0, 'mix': 0}
    les_miss_ids = {'cal': [], 'low': [], 'mix': []}
    find_gid_dict = {gid: True for gid in eval_info['find_indices']}
    for gid, gt_roi_dict in enumerate(eval_info['gt_info']['roi_dicts']):
        les_type = gt_roi_dict['type']
        if gid in find_gid_dict:
            les_find_nums[les_type] += 1
        else:
            les_miss_ids[les_type].append(gid)
        les_gt_nums[les_type] += 1
    title = '%s: ' % name
    for les_type in ['cal', 'low', 'mix']:
        miss_ids_str = '(%s)' % '|'.join([str(i) for i in les_miss_ids[les_type]]) if len(les_miss_ids[les_type]) > 0 else ''
        title += '%s=%d/%d%s, ' % (les_type[:1], les_find_nums[les_type], les_gt_nums[les_type], miss_ids_str)
    title += 'FP=%d' % (len(eval_info['pred_bbox_list']) - sum(eval_info['pred_correct_flags']))
    img_rgb = draw_pred_list(img_rgb, eval_info['pred_bbox_list'])
    draw_title(img_rgb, title)
    return img_rgb


def load_and_pad_image(img_path):
    img = load_image(img_path)
    img = numpy.vstack([numpy.zeros([PAD_Y, img.shape[1]], dtype='uint8'), img])
    img_h, img_w = img.shape
    return img.reshape([1, img_h, img_w])


def draw_images(pssi_pred_dict, pssi_all_eval_info, pssi_eval_info, pssi_merged_eval_info):
    # origin |    rgb     | gt
    # detect | det/filter | merge
    pssi_dataset_dict = load_pssi_dataset_name()
    pssi_list = sorted(pssi_pred_dict.keys())
    save_dir = '/breast_data/zhangfd/cta/ai_results/200229/model_vis_ana/'
    for pssi in pssi_list:
        # if len(pssi_all_eval_info[pssi]['gt_info']['roi_dicts']) == 0:
        #     continue
        sub_img_path = '%s/%s.png' % (pssi_dataset_dict[pssi], pssi)
        img_ann = load_and_pad_image('/breast_data/cta/image/-100_700/' + sub_img_path)
        img_400_1000 = load_and_pad_image('/breast_data/cta/image/400_1000/' + sub_img_path)
        img_0_600 = load_and_pad_image('/breast_data/cta/image/0_600/' + sub_img_path)
        img_min_max = load_and_pad_image('/breast_data/cta/image/min_max/' + sub_img_path)
        img_ann_rgb = gray2rgb(img_ann)
        img_train_rgb = numpy.vstack([img_400_1000, img_0_600, img_min_max]).transpose(1, 2, 0)
        img_gt_rgb = draw_gt_image(img_ann_rgb.copy(), pssi_eval_info[pssi]['gt_info'])
        canvas_top = numpy.hstack([img_ann_rgb, img_train_rgb, img_gt_rgb])
        img_all_pred = draw_pred_image(img_ann_rgb.copy(), pssi_all_eval_info[pssi], pssi_all_eval_info['name'])
        img_filtered_pred = draw_pred_image(img_ann_rgb.copy(), pssi_eval_info[pssi], pssi_eval_info['name'])
        img_merged_pred = draw_pred_image(img_ann_rgb.copy(), pssi_merged_eval_info[pssi], pssi_merged_eval_info['name'])
        canvas_bottom = numpy.hstack([img_all_pred, img_filtered_pred, img_merged_pred])
        # draw_img = numpy.vstack([canvas_top, canvas_bottom])
        canvas1 = numpy.hstack([img_ann_rgb, img_train_rgb])
        canvas2 = numpy.hstack([img_gt_rgb, img_all_pred])
        canvas3 = numpy.hstack([img_filtered_pred, img_merged_pred])
        draw_img = numpy.vstack([canvas1, canvas2, canvas3])
        mkdir_safe(save_dir + pssi_dataset_dict[pssi])
        save_image(save_dir + sub_img_path, draw_img)
        # break


def main():
    iou_thresh, score_thresh = float(sys.argv[1]), float(sys.argv[2])
    # iou_thresh, score_thresh = 0.1, 0.3
    dataset_names = ['test00_test']
    dataset_names += ['train01_test']
    evaluator = PlaqueEvaluator('+'.join(dataset_names))
    evaluator.load_gt(load_gt_jsons())
    pred_base_dir = '/breast_data/zhangqianyi/models/cta_200217_mask_cas/inference/model_0050000/'
    pssi_pred_dict = {}
    for dataset_name in dataset_names:
        cur_pred_json = json.load(open(pred_base_dir + 'cta_%s/cta_%s.json' % (dataset_name, dataset_name)))
        if dataset_name == 'train01_test':
            pssi_pop_list = []
            for pssi in cur_pred_json.keys():
                if '0657701' <= pssi.split('_')[0] <= '0657726':
                    pssi_pop_list.append(pssi)
            for pssi in pssi_pop_list:
                _ = cur_pred_json.pop(pssi)
        pssi_pred_dict.update(cur_pred_json)
    # pssi_all_eval_info = evaluator.evaluate(pssi_pred_dict, 0.1, 'iou', iou_thresh, False, is_verbose=False)
    # pssi_all_eval_info['name'] = 'IOU%s/s0.1' % iou_thresh
    # print('')
    # pssi_eval_info = evaluator.evaluate(pssi_pred_dict, score_thresh, 'iou', iou_thresh, False, is_verbose=False)
    # pssi_eval_info['name'] = 'IOU%s/s%s' % (iou_thresh, score_thresh)
    # print('')
    # _ = evaluator.evaluate(pssi_pred_dict, score_thresh, 'iou', iou_thresh, True, is_verbose=False)
    # print('')
    # _ = evaluator.evaluate(pssi_pred_dict, score_thresh, 'iomin', iou_thresh, False, is_verbose=False)
    # print('')
    pid_list = sorted(list(set([pssi.split('_')[0] for pssi in pssi_pred_dict.keys()])))
    save_string_list('/breast_data/cta/new_data/b1_sg_587/config/old/test_pid_list.txt', pid_list)
    pssi_merged_eval_info = evaluator.evaluate(pssi_pred_dict, score_thresh, 'iomin', iou_thresh, True, is_verbose=False)
    pssi_merged_eval_info['name'] = 'MG/IOM%s/s%s' % (iou_thresh, score_thresh)
    # draw_images(pssi_pred_dict, pssi_all_eval_info, pssi_eval_info, pssi_merged_eval_info)


if __name__ == '__main__':
    # gen_images()
    main()

