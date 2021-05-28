from seg_util import *
import nibabel as nib
import shutil
from skimage import measure
from concurrent.futures import ProcessPoolExecutor
from lib.utils.mio import mkdir_safe, save_image, load_string_list
import sys
sys.path.insert(0, '/breast_data/zhangfd/code/ccta/torchseg_v2/torchseg/')
from metrics.eval_vessel import eval_ccta_volume, eval_ccta_volume_v2, post_process_v2


def get_mask_tensor(mask_tensor, image_tensor):
    img_mask_tensor = np.uint8(image_tensor * 0.4)
    img_mask_tensor[mask_tensor == 3] = 255
    img_mask_tensor[mask_tensor == 2] = 208
    img_mask_tensor[mask_tensor == 1] = 160
    return img_mask_tensor


def draw_volume(image_tensor, mask_tensor, heatmap_tensor, pred_tensors=None):
    heatmap_tensor = np.uint8(heatmap_tensor * 127)
    # mask_tensor[mask_tensor > 0] = 1
    # img_mask_tensor = np.uint8(image_tensor * 0.5)
    # img_mask_tensor[mask_tensor > 0] = 255
    img_mask_tensor = get_mask_tensor(mask_tensor, image_tensor)
    draw_tensor = np.concatenate([image_tensor, img_mask_tensor, heatmap_tensor], axis=2)
    if pred_tensors is not None:
        row_tensor2 = np.concatenate([get_mask_tensor(m, image_tensor) for m in pred_tensors], axis=2)
        draw_tensor = np.concatenate([draw_tensor, row_tensor2], axis=1)
    # print(draw_tensor.shape, mask_tensor.sum(axis=1).sum(axis=1), heatmap_tensor.sum())
    return draw_tensor


def draw_random_patches(image, mask, heatmap, skemask, pred_masks, patch_num, save_dir, patch_size):
    for sample_id in range(patch_num):
        tensors = [heatmap, skemask] + pred_masks
        img_patch, mask_patch, other_patches, param = sample_one_patch(image, mask, tensors, patch_size)
        # print(img_patch.shape, mask_patch.shape, other_patches[0].shape)
        x, y, z, flip_y, flip_z = param
        save_dir_i = '%s_%d_%d_%d_%s_%s/' % (save_dir, x, y, z, flip_y, flip_z)
        mkdir_safe(save_dir_i)
        draw_tensor = draw_volume(img_patch, mask_patch, other_patches[0], other_patches[2:])
        for i in range(0, draw_tensor.shape[0], 2):
            save_image(save_dir_i + '%04d.png' % (i + x), draw_tensor[i])


def draw_main():
    base_dir = '/home/data/ccta/'
    save_dir = base_dir + 'draw/gt_hp_sk_p8/'
    valid_pids = sorted(load_string_list('/brain_data/dataset/ccta/validate_all_subjects_20200410.lst'))
    for i, valid_pid in enumerate(valid_pids):
        image, mask, heatmap, skemask = load_vessel_seg_data(base_dir, valid_pid)
        draw_random_patches(image, mask, heatmap, skemask, [], 8, save_dir + valid_pid + '/', (32, 384, 384))
        print('No.%d/%d: %s' % (i, len(valid_pids), valid_pid))


def compute_dice(gt, seg):
    n_gt = np.sum(gt > 0)
    n_seg = np.sum(seg > 0)
    n_union = np.sum((gt + seg) > 0)
    n_inter = n_gt + n_seg - n_union
    dice = 2.0 * n_inter / (n_gt + n_seg)
    prec = 1.0 * n_inter / n_seg
    recall = 1.0 * n_inter / n_gt
    raw_dice = [recall, prec, dice]
    return raw_dice


def get_eval_result(seg, gt, num_class):
    true_gt = gt.copy()
    true_gt[gt >= num_class] = (num_class - 1)
    result = eval_ccta_volume(seg, true_gt, num_class)
    return compute_dice(true_gt, seg) + result


def format_eval_str(result):
    recall, prec, dice = result[:3]
    s = 'raw_dice: r=%.4f/p=%.4f/d=%.4f, ' % (recall, prec, dice)
    cls_dices = result[3:-4]
    vol_dice, hd, assd, clcr = result[-4:]
    vdclcr = vol_dice * clcr
    s += 'dice=%.4f(%s), ' % (vol_dice, '/'.join(['%.4f' % d for d in cls_dices]))
    s += 'hd=%.4f, assd=%.4f, clcr=%.4f, vdclcr=%.4f' % (hd, assd, clcr, vdclcr)
    return s


def compute_dice_v2(gt, seg):
    seg_mask, gt_mask = seg > 0, gt > 0
    n_gt = np.sum(gt_mask)
    n_seg = np.sum(seg_mask)
    n_union = np.sum((gt + seg) > 0)
    n_inter = n_gt + n_seg - n_union
    dice = 2.0 * n_inter / (n_gt + n_seg)
    prec = 1.0 * n_inter / n_seg
    recall = 1.0 * n_inter / n_gt
    # recall of gt > 1
    gt2_mask = gt > 1
    recall2 = np.sum(gt2_mask * seg_mask) * 1. / gt2_mask.sum()
    # precison of seg > 1
    seg2_mask = seg > 1
    prec2 = np.sum(seg2_mask * gt_mask) * 1. / seg2_mask.sum()
    raw_dice = [recall, prec, dice, recall2, prec2]
    return raw_dice


def get_eval_result_v2(seg, gt, num_class):
    true_gt = gt.copy()
    true_gt[gt >= num_class] = (num_class - 1)
    result = eval_ccta_volume_v2(seg, true_gt, num_class)
    return compute_dice_v2(true_gt, seg) + result


def format_eval_str_v2(result):
    recall, prec, dice, recall2, prec2 = result[:5]
    cls_dices = result[5:-10]
    vol_dice, recall2, prec2, sr, sp, sr1, sp1, sr2, sp2, clcr = result[-10:]
    s = 'raw_dice: r=%.4f/p=%.4f/d=%.4f/r2=%.4f/p2=%.4f, ' % (recall, prec, dice, recall2, prec2)
    s += 'dice=%.4f(%s), ' % (vol_dice, '/'.join(['%.4f' % d for d in cls_dices]))
    s += 'sr/sp=%.4f/%.4f, sr1/sp1=%.4f/%.4f, sr2/sp2=%.4f/%.4f, clcr=%.4f' % (sr, sp, sr1, sp1, sr2, sp2, clcr)
    return s


def gen_eval_results():
    base_dir = '/breast_data/cta/new_data/vessel_seg/with_sg_407_plaque/'
    # mask_dir = base_dir + 'mask_wp4/'
    mask_dir = '/brain_data/dataset/ccta/mask_all3/'
    base_model_dir = '/home/zhangfd/code/ccta/torchseg_v2/torchseg/results/ccta_vessel/train/'
    model_dirs = [
        # base_model_dir + 'da_seresnet18_skel_rot_mix_loss_all_class3_20200425/best_VDCLCR_pred/',
        # base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_20200506/best_VDCLCR_pred/',
        # base_model_dir + 'wp407/da_seresnet18c24_c4_w100_skel_hp_mix_loss_wp407_20200506/best_VDCLCR_pred/'
        base_model_dir + 'p128x512x512/from_pcw/',
        base_model_dir + 'da_seresnet18_skel_rot_mix_loss_all_class3_20200425/best_VDCLCR_pred/valid50/',
        base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_20200506/best_VDCLCR_pred/valid50/',
        base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_dlt2pix_20200511/best_F-SRp7SP/valid50/'
    ]
    # num_classes = [3, 3, 4]
    num_classes = [3, 3, 3, 3]
    # subjects = load_string_list(base_dir + 'config/valid_test_list.txt')
    subjects = load_string_list('/brain_data/dataset/ccta/validate_all_subjects_20200410.lst')
    all_eval_results = []
    # model_dirs, num_classes = model_dirs[1:], num_classes[1:]
    for model_dir, num_class in zip(model_dirs, num_classes):
        eval_npy_path = model_dir + 'eval_v2.npy'
        if os.path.exists(eval_npy_path):
            eval_results = np.load(eval_npy_path)
            print('loaded %s' % eval_npy_path)
        else:
            eval_results = []
            print('evaluating %s' % model_dir)
            ex = ProcessPoolExecutor()
            for i, subject in enumerate(subjects):
                mask = nib.load(get_nii_data_path(mask_dir, subject))
                mask = np.uint8(mask.get_data())
                gt = np.transpose(mask, (2, 1, 0))
                seg = np.transpose(nib.load(model_dir + '%s_seg.nii.gz' % subject).get_data(), (2, 1, 0))
                future = ex.submit(get_eval_result_v2, seg, gt, num_class)
                eval_results.append(future)
            ex.shutdown(wait=True)
            eval_results = np.float32([result.result() for result in eval_results])
            np.save(eval_npy_path, eval_results)
            print('saved %s' % eval_npy_path)
        all_eval_results.append(eval_results)
    for i, subject in enumerate(subjects):
        is_skip = False
        for model_id, (eval_results, model_dir) in enumerate(zip(all_eval_results, model_dirs)):
            delta = eval_results[i, -7] - eval_results[i, -5]
            break_str = ' BREAK=%.4f' % delta if delta > 0.01 else ''
            if eval_results[i, -7] < 0.5:
                is_skip = True
            # if is_skip:
            #     continue
            print('MODEL %d: No.%d/%d, %s: %s%s' % (
                model_id, i + 1, len(subjects), subject, format_eval_str_v2(eval_results[i]), break_str))
        # if not is_skip:
        print('')
    num_subjects = len(subjects)
    for eval_results, model_dir in zip(all_eval_results, model_dirs):
        print(model_dir)
        sr_array, sr1_array = eval_results[:, -7], eval_results[:, -5]
        break_num_01 = int(np.sum((sr_array - sr1_array) > 0.01))
        break_num_02 = int(np.sum((sr_array - sr1_array) > 0.02))
        sr_a95, sr_a90 = np.sum(sr_array > 0.95) * 100. / num_subjects, np.sum(sr_array > 0.9) * 100. / num_subjects
        sr1_a95, sr1_a90 = np.sum(sr1_array > 0.95) * 100. / num_subjects, np.sum(sr1_array > 0.9) * 100. / num_subjects
        eva_str = format_eval_str_v2(eval_results.mean(axis=0))
        print('\t%s, Break01=%d, Break02=%d, [SR>95%%]=%.2f%%, [SR>90%%]=%.2f%%, [SR1>95%%]=%.2f%%, [SR1>90%%]=%.2f%%' % (
            eva_str, break_num_01, break_num_02, sr_a95, sr_a90, sr1_a95, sr1_a90))


def draw_pred_cmp_main():
    base_dir = '/breast_data/cta/new_data/vessel_seg/with_sg_407_plaque/'
    mask_dir = base_dir + 'mask_wp4/'
    image_dir = '/home/data/ccta/image/'
    heatmap_dir = '/brain_data/dataset/ccta/heatmap/'
    ske_dir = '/brain_data/dataset/ccta/skeleton/'
    save_dir = base_dir + 'draw/cmp_hp_wp3_wp4_v2/'
    base_model_dir = '/home/zhangfd/code/ccta/torchseg_v2/torchseg/results/ccta_vessel/train/'
    model_dirs = [
        base_model_dir + 'da_seresnet18_skel_rot_mix_loss_all_class3_20200425/best_VDCLCR_pred/',
        base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_20200506/best_VDCLCR_pred/',
        base_model_dir + 'wp407/da_seresnet18c24_c4_w100_skel_hp_mix_loss_wp407_20200506/best_VDCLCR_pred/'
    ]
    num_classes = [3, 3, 4]
    subjects = load_string_list(base_dir + 'config/valid_test_list.txt')
    all_eval_results = [np.load(model_dir + 'eval.npy') for model_dir in model_dirs]
    for i, subject in enumerate(subjects):
        img_nii = nib.load(get_nii_data_path(image_dir, subject))
        image = img_nii.get_data()
        mask = nib.load(get_nii_data_path(mask_dir, subject))
        mask = mask.get_data()
        image = np.transpose(image, (2, 1, 0))
        mask = np.transpose(mask, (2, 1, 0))
        heatmap = nib.load(get_nii_data_path(heatmap_dir, subject))
        heatmap = heatmap.get_data().astype('float32')
        heatmap = np.transpose(heatmap, (2, 1, 0))
        points = scio.loadmat(ske_dir + '%s.mat' % subject)['points']
        points = points[:, ::-1]
        skemask = np.ones_like(mask) * 127
        psf_points = psf(points, 2, mask.shape, as_tuple=True)
        skemask[psf_points] = mask[psf_points] * 255
        pred_tensors = [
            np.transpose(nib.load(md + '%s_seg.nii.gz' % subject).get_data(), (2, 1, 0)) for md in model_dirs]
        save_dir_i = save_dir + subject + '/'
        draw_random_patches(image, mask, heatmap, skemask, pred_tensors, 8, save_dir_i, [64, 384, 384])
        image_wl_ww = set_window_wl_ww(image, 400, 1000)
        draw_tensor = draw_volume(image_wl_ww, mask, heatmap, pred_tensors)
        all_save_dir_i = save_dir_i + 'all/'
        mkdir_safe(all_save_dir_i)
        for z in range(0, draw_tensor.shape[0], 2):
            save_image(all_save_dir_i + '%04d.png' % z, draw_tensor[z])
        # dice_list = [compute_dice(mask, pm) for pm in pred_tensors]
        # eval_results = [get_eval_result(seg, mask, num_class) for seg, num_class in zip(pred_tensors, num_classes)]
        # all_eval_results.append(eval_results)
        print('No.%d/%d: %s' % (i, len(subjects), subject))
        for model_id, eval_results in enumerate(all_eval_results):
            # all_eval_results[model_id].append(eval_result)
            eval_str = format_eval_str(eval_results[i])
            # average_eval_str = format_eval_str(np.float32(all_eval_results[model_id]).mean(axis=0))
            # print('\tnum_class=%d. %s. average: %s' % (num_classes[model_id], eval_str, average_eval_str))
            print('\tnum_class=%d. %s.' % (num_classes[model_id], eval_str))


def post_process(seg):
    labels = measure.label(seg)
    props = measure.regionprops(labels)
    for prop in props:
        if prop.bbox[5] - prop.bbox[2] < 20:
            seg[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = 0
    return seg


def gen_model_seg_cmp_nii():
    base_dir = '/breast_data/cta/new_data/vessel_seg/with_sg_407_plaque/'
    # mask_dir = base_dir + 'mask_wp4/'
    mask_dir = '/brain_data/dataset/ccta/mask_all3/'
    # save_dir = base_dir + 'draw/cmp_wp3_dlt1pix_nii/'
    save_dir = base_dir + 'draw/cmp_hsm_nii/'
    base_model_dir = '/home/zhangfd/code/ccta/torchseg_v2/torchseg/results/ccta_vessel/train/new_train/'
    model_dirs = [
        # base_model_dir + 'da_seresnet18_skel_rot_mix_loss_all_class3_20200425/best_VDCLCR_pred/',
        # base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_20200506/best_VDCLCR_pred/',
        # base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_dlt1pix_20200511/best_F-SRp3SP/',
        # base_model_dir + 'wp407/da_seresnet18c24_c3_skel_hp_mix_loss_wp407_dlt2pix_20200511/best_F-SRp3SP/'
        base_model_dir + 'da_seres18c24_32x384x384_20200513/best_F-SRp7SP/valid50/',
        base_model_dir + 'da_seres18c24_32x384x384_20200515_hkm10/best_F-SRp7SP/valid50/',
    ]
    # subjects = load_string_list(base_dir + 'config/valid_test_list.txt')
    subjects = load_string_list('/brain_data/dataset/ccta/validate_all_subjects_20200410.lst')
    mkdir_safe(save_dir)
    for i, subject in enumerate(subjects):
        mask = nib.load(get_nii_data_path(mask_dir, subject))
        gt = mask.get_data()
        seg1, seg2 = [nib.load(md + '%s_seg.nii.gz' % subject).get_data() for md in model_dirs]
        gt = np.uint8(gt > 0)
        seg1 = np.uint8(post_process_v2(seg1 > 0)[0]) * 2
        seg2 = np.uint8(post_process_v2(seg2 > 0)[1]) * 4
        seg_cmp = gt + seg1 + seg2
        nib.save(nib.Nifti1Image(seg_cmp, mask.affine), save_dir + '%s.nii.gz' % subject)
        seg_label_str = '/'.join(['%d=%d' % (v, int(np.sum(seg_cmp == v))) for v in range(1, 8)])
        print('No.%d/%d, %s, %s' % (i, len(subjects), subject, seg_label_str))


def copy_image_nii():
    base_dir = '/breast_data/cta/new_data/vessel_seg/with_sg_407_plaque/'
    img_dir = '/brain_data/dataset/ccta/image/'
    save_dir = base_dir + 'image_nii_gz/'
    mkdir_safe(save_dir)
    subjects = load_string_list(base_dir + 'config/valid_test_list.txt')
    for i, subject in enumerate(subjects):
        shutil.copy(img_dir + '%s.nii.gz' % subject, save_dir + '%s.nii.gz' % subject)


if __name__ == '__main__':
    # draw_main()
    # draw_pred_cmp_main()
    # gen_eval_results()
    gen_model_seg_cmp_nii()
    # copy_image_nii()

