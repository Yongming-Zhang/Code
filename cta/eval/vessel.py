import numpy as np
from scipy import ndimage
from skimage import measure
from functools import reduce
from skimage.morphology import skeletonize_3d, binary_opening, binary_closing, binary_dilation, ball


def dice(seg_bin, gt_bin):
    return 2.0 * (seg_bin * gt_bin).sum() / (seg_bin.sum() + gt_bin.sum())


def skeleton(mask):
    volume = skeletonize_3d(mask)
    points = np.asarray(np.where(volume == True)).T
    return points


def overlapping(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return 0
    w = max(np.max(P[:, 0]), np.max(Q[:, 0])) + 10
    h = max(np.max(P[:, 1]), np.max(Q[:, 1])) + 10
    d = max(np.max(P[:, 2]), np.max(Q[:, 2])) + 10

    vq = np.zeros((w, h, d), 'uint8')
    vq[Q[:, 0], Q[:, 1], Q[:, 2]] = 1
    vp = np.zeros_like(vq)
    vp[P[:, 0], P[:, 1], P[:, 2]] = 1

    match = 1
    vv = np.zeros_like(vp)
    for p in P:
        x, y, z = p
        if vq[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2].sum() > 0:
            match = match + 1
        else:
            vv[x, y, z] = 1
    return 1.0 * match / len(P)


def split_vessel(mask):
    # step two: zoom data and erosion
    scale_factor = 0.125
    src_shape = np.array(mask.shape, dtype=float)
    dst_shape = (src_shape * scale_factor).astype('int32').astype('float')
    nmask = ndimage.zoom(mask, np.divide(dst_shape, src_shape), order=0)
    cleaned_nmask = binary_opening(nmask > 0, ball(1))

    # step three: dilation and zoom data
    cleaned_nmask = binary_dilation(cleaned_nmask, ball(1))
    cleaned_nmask = binary_dilation(cleaned_nmask, ball(1))
    cleaned_nmask_enlarged = ndimage.zoom(cleaned_nmask, np.divide(src_shape, dst_shape), order=0)
    cleaned_nmask_enlarged = binary_dilation(cleaned_nmask_enlarged, ball(1))

    # step four: get thin vessel mask
    main_vessel_mask = np.logical_and(mask, cleaned_nmask_enlarged > 0).astype('uint8')
    labels, ncomponents = ndimage.measurements.label(main_vessel_mask, np.ones((3, 3, 3), dtype=np.int))
    if labels.max() != 0:
        main_vessel_mask = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    thin_vessel_mask = mask - main_vessel_mask
    thin_vessel_mask[thin_vessel_mask < 0] = 0

    labels = measure.label(thin_vessel_mask)
    props = measure.regionprops(labels)
    for prop in props:
        if prop.bbox[3] - prop.bbox[0] < 8 or prop.bbox[4] - prop.bbox[1] < 8 or prop.bbox[5] - prop.bbox[2] < 16:
            thin_vessel_mask[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = 0

    # step five: use axial to refine main vessel mask
    main_vessel_mask = mask - thin_vessel_mask
    for z in range(mask.shape[2]):
        if main_vessel_mask[:, :, z].sum() == 0:
            continue
        labels = measure.label(main_vessel_mask[:, :, z])
        props = measure.regionprops(labels)
        prop = reduce(lambda x, y: x if x['area'] > y['area'] else y, props)
        area = prop.area
        for prop in props:
            if 1.0 * prop.area / area < 0.05:
                main_vessel_mask[prop.coords[:, 0], prop.coords[:, 1], z] = 0

    # step six: get final vessel mask
    thin_vessel_mask = mask - main_vessel_mask
    labels = measure.label(thin_vessel_mask)
    props = measure.regionprops(labels)
    for prop in props:
        if prop.bbox[3] - prop.bbox[0] < 8 or prop.bbox[4] - prop.bbox[1] < 8 or prop.bbox[5] - prop.bbox[2] < 16:
            thin_vessel_mask[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = 0
    main_vessel_mask = mask - thin_vessel_mask

    return main_vessel_mask, thin_vessel_mask


def skeleton_correct_rate(ske, correct_mask):
    correct_num = filter_pts_by_mask(ske, correct_mask).shape[0]
    return correct_num * 1.0 / (ske.shape[0] + 1e-5)


def filter_pts_by_mask(pts, mask):
    return np.array(list(filter(lambda x: mask[x[0], x[1], x[2]] > 0, pts)))


def get_largest_connected_region(mask):
    labels = measure.label(mask > 0)
    props = measure.regionprops(labels)
    area_labels = []
    for prop in props:
        area_labels.append([prop.area, prop.label])
    area_labels.sort()
    return labels == area_labels[-1][1]


def eval_segments(pred, gt, gt_segments_pts, gt_segments_types, min_seg_recalls, min_seg_pts=20):
    seg_types_recalls, sum_recall = [], 0
    sum_bridge_recall, bridge_num = 0, 0
    sum_end_recall, end_num = 0, 0
    for seg_pts, seg_type in zip(gt_segments_pts, gt_segments_types):
        seg_pts = np.array(list(filter(lambda x: gt[x[0], x[1], x[2]] != 3, seg_pts)))
        if seg_pts.shape[0] < min_seg_pts:
            continue
        seg_recall = skeleton_correct_rate(seg_pts, pred)
        seg_types_recalls.append([seg_type, seg_recall])
        sum_recall += seg_recall
        if seg_type == 'bridge':
            sum_bridge_recall += seg_recall
            bridge_num += 1
        else:
            sum_end_recall += seg_recall
            end_num += 1
    num_segments = len(seg_types_recalls)
    avg_recall = sum_recall / (num_segments + 1e-5)
    avg_bridge_recall = sum_bridge_recall / (bridge_num + 1e-5)
    avg_end_recall = sum_end_recall / (end_num + 1e-5)
    results = [avg_recall, avg_bridge_recall, avg_end_recall]
    for min_seg_recall in min_seg_recalls:
        num_recall, bridge_num_recall, end_num_recall = 0, 0, 0
        for seg_type, seg_recall in seg_types_recalls:
            if seg_recall >= min_seg_recall:
                num_recall += 1
                if seg_type == 'bridge':
                    bridge_num_recall += 1
                else:
                    end_num_recall += 1
        total_recall = num_recall / (num_segments + 1e-5)
        bridge_recall = bridge_num_recall / (bridge_num + 1e-5)
        end_recall = end_num_recall / (end_num + 1e-5)
        results += [total_recall, bridge_recall, end_recall]
    return results


def eval_skeleton(pred, gt, gt_ske_pts, gt_segments_pts, gt_segments_types):
    # gt_ignore_ske_pts = np.array(list(filter(lambda x: gt[x[0], x[1], x[2]] == 3, gt_ske_pts)))
    gt_req_ske_pts = np.array(list(filter(lambda x: gt[x[0], x[1], x[2]] != 3, gt_ske_pts)))
    pred_ske_pts = skeleton(pred)
    gt_all_mask = (gt > 0).astype('uint8')
    _, gt_thin_vessel = split_vessel(gt_all_mask)
    _, pred_thin_vessel = split_vessel(pred)
    cr = overlapping(
        filter_pts_by_mask(gt_req_ske_pts, gt_thin_vessel), filter_pts_by_mask(pred_ske_pts, pred_thin_vessel))
    sr = skeleton_correct_rate(gt_ske_pts, pred)
    sr_req = skeleton_correct_rate(gt_req_ske_pts, pred)
    sp = skeleton_correct_rate(pred_ske_pts, gt_all_mask)
    seg_recalls = eval_segments(pred, gt, gt_segments_pts, gt_segments_types, [0.7, 0.8, 0.9])
    # top1_pred = get_largest_connected_region(pred)
    # sr1 = skeleton_correct_rate(gt_ske_pts, top1_pred)
    # sp1 = skeleton_correct_rate(filter_pts_by_mask(pred_ske_pts, top1_pred), gt)
    return seg_recalls + [sr, sr_req, sp, cr]


def eval_vessel(seg, gt_dict):
    gt = gt_dict['gt']
    gt_ske_pts = gt_dict['ske_pts']
    gt_segments_pts = gt_dict['seg_pts']
    gt_segments_types = gt_dict['seg_types']
    # pixel-level accuracy
    results = []
    seg_mask, gt_mask = seg > 0, gt > 0
    gt_ignore_mask = (gt == 3)
    gt_required_mask = gt_mask * (gt != 3)
    inter = np.sum(seg_mask * gt_mask)
    inter_req = np.sum(seg_mask * gt_required_mask)
    sum_seg, sum_gt, sum_gt_req = seg_mask.sum(), gt_mask.sum(), gt_required_mask.sum()
    dice_all = 2. * inter / (sum_seg + sum_gt)
    dice_required = 2. * inter_req / (sum_seg + sum_gt_req)
    pixel_all_recall = inter * 1. / sum_gt
    pixel_ignore_recall = np.sum(seg_mask * gt_ignore_mask) * 1. / gt_ignore_mask.sum()
    pixel_required_recall = inter_req * 1. / sum_gt_req
    pixel_prec = inter * 1. / sum_seg
    results += [dice_all, dice_required, pixel_all_recall, pixel_ignore_recall, pixel_required_recall, pixel_prec]

    # skeleton-level
    ske_results = eval_skeleton(seg_mask.astype('uint8'), gt, gt_ske_pts, gt_segments_pts, gt_segments_types)
    clcr = dice_required * ske_results[-1]
    ske_results.append(clcr)
    results += ske_results

    return results
