from cta_util import *
import cv2
import glob
import shutil
from collections import defaultdict
from lib.utils.mio import *
from lib.image.image import gray2rgb
from lib.image.draw import draw_texts


def get_pids(base_dir):
    old_new_path = base_dir + 'config/old_new_map.txt'
    new_old_path = base_dir + 'config/new_old_map.txt'
    lines, pids, new_index = [], [], 0
    if os.path.exists(old_new_path):
        lines = load_string_list(old_new_path)
        new_index = 1
    elif os.path.exists(new_old_path):
        lines = load_string_list(new_old_path)
    for line in lines:
        pids.append(line.split(' ')[new_index])
    if len(pids) == 0:
        sub_dirs = load_string_list(base_dir + 'config/sub_dirs.txt')
        pids = [sub_dir.split('_')[0] for sub_dir in sub_dirs]
    return sorted(pids)


def load_all_centerline(centerline_dir):
    c2d_dict = {}
    for c2d_path in sorted(glob.glob(centerline_dir + '*.2d')):
        c2ds = json_load(c2d_path)
        for c2d in c2ds['vessel_group']:
            c2d_dict[c2d['name']] = numpy.float32(c2d['points'])
    return c2d_dict


def draw_c2d(img, c2d_pts, cpr_mask=None):
    img = gray2rgb(img) if cpr_mask is None else proc_mask(img, cpr_mask)
    c2d_pts = c2d_pts.round().astype('int32')
    for i in range(c2d_pts.shape[0] - 1):
        x0, y0 = c2d_pts[i]
        x1, y1 = c2d_pts[i + 1]
        cv2.line(img, (x0, y0), (x1, y1), (255, 255, 0), 1)
    return img


def parse_ai_json(ai_jd):
    vessel_slice_dict = defaultdict(list)
    nodes = [('plaque', plaque) for plaque in ai_jd['plaque']]
    nodes += [('stent', stent) for stent in ai_jd['stent']]
    for i, (node_type, node) in enumerate(nodes):
        for ep_id, ep in enumerate(node['end_points']):
            vessel_slice_id = node['position'] + '_%d' % (ep['slice_index'] - 1)
            d = {
                'type': node_type,
                'index': i,
                'segment': node['segment'],
                'end_points': ep['edge'],
                'num_nodes': len(node.get('det_nodes', [])),
                'score': node.get('score', 0),
                'is_major': ep_id == 0
            }
            if node_type == 'plaque':
                d['plaque_info'] = [
                    node['plaque_type'], node['stenosis_type'], node['plaque_distribution'],
                    node['stenosis_rate'], node['stenosis_r1'], node['stenosis_r2']
                ]
            vessel_slice_dict[vessel_slice_id].append(d)
    return vessel_slice_dict


def draw_slice(img, nodes):
    img = gray2rgb(img)
    plaque_color_dict = {
        'noncalcified': (0, 0, 255),
        'calcified': (0, 255, 0),
        'mix': (0, 97, 255),
    }
    txt_color_dict = {
        'none': (112, 220, 240),
        'lower': (110, 241, 252),
        'low': (0, 212, 255),
        'mid': (32, 121, 244),
        'high': (60, 107, 248),
        'total': (0, 0, 255),
    }
    pad, radius = 10, 15
    for node in nodes:
        txts = ['No.%d/s=%.2f/%d/%s' % (node['index'], node['score'], node['num_nodes'], node['segment'])]
        is_bold = node['is_major']
        thick = int(is_bold) + 1
        (x0, y0), (xc, yc), (x1, y1) = node['end_points']
        xs = [x0 - pad, x0 + pad, x1 - pad, x1 + pad]
        ys = [y0 - pad, y0 + pad, y1 - pad, y1 + pad]
        bx0, bx1, by0, by1 = int(round(min(xs))), int(round(max(xs))), int(round(min(ys))), int(round(max(ys)))
        xc, yc = int(round(xc)), int(round(yc))
        if node['type'] == 'plaque':
            plaque_type, stenosis_type, dist, rate, r1, r2 = node['plaque_info']
            color = plaque_color_dict[plaque_type]
            txt_color = txt_color_dict[stenosis_type]
            plaque_type_short = 'low' if plaque_type == 'noncalcified' else plaque_type[:3]
            txts += [
                '%s/%s/r=%.2f%%' % (plaque_type_short, dist[:3], rate * 100),
                '%.2f/%.2fmm' % (r1, r2)
            ]
            cv2.circle(img, (xc, yc), radius, color, thickness=thick)
            draw_texts(img, (xc - radius, yc - radius), ['No.%d' % node['index']], txt_color, direct=1, thick=thick, font_scale=0.5)
        else:
            color = (255, 0, 0)
            txt_color = color
        draw_texts(img, (bx0, by0), txts, txt_color, direct=1, thick=thick, font_scale=0.5)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), color=color, thickness=thick)
    return img


def proc_mask(img, mask):
    img_mask = numpy.uint8(img * 0.5)
    img_mask[mask > 0] = 255
    return gray2rgb(img_mask)


def draw_image(img, cpr_mask, ai_nodes, c2d_pts, det_imgs, masks):
    img_org = gray2rgb(img)
    img_c2d = draw_c2d(img, c2d_pts, cpr_mask)
    img_ai = draw_slice(img, ai_nodes)
    masks = [proc_mask(img, mask) for mask in masks]
    draw_imgs = [img_org, img_c2d, img_ai] + det_imgs[3:] + masks
    if len(draw_imgs) % 2 == 1:
        draw_imgs.append(numpy.zeros(img_org.shape, dtype='uint8'))
    if True:
        h, w = img.shape
        pad = 96
        x0, y0 = c2d_pts.min(axis=0)
        x1, y1 = c2d_pts.max(axis=0)
        bx0 = max(0, int(round(x0 - pad)))
        by0 = max(0, int(round(y0 - pad)))
        bx1 = min(w, int(round(x1 + pad)))
        by1 = min(h, int(round(y1 + pad)))
        draw_imgs = [dimg[by0:by1, bx0:bx1] for dimg in draw_imgs]
    half_len = len(draw_imgs) // 2
    draw_img = numpy.vstack([
        numpy.hstack(draw_imgs[:half_len]),
        numpy.hstack(draw_imgs[half_len:])
    ])
    return draw_img


def load_det_imgs(det_dir, file_name):
    img_path = det_dir + file_name + '.png'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        h2, w2 = img.shape[:2]
        h, w = h2 // 2, w2 // 2
        return [img[:h, :w], img[:h, w:], img[h:, :w], img[h:, w:]]
    return []


def load_masks(mask_dir, file_name):
    img_path = mask_dir + file_name + '.png'
    if os.path.exists(img_path):
        img = load_image(img_path)
        h2, w = img.shape[:2]
        h = h2 // 2
        return [img[:h], img[h:]]
    return []


def draw_ai(ai_dir, save_dir):
    if not os.path.exists(ai_dir + 'diagnose_debug/diag_result.json'):
        return False
    dcm_paths = sorted(glob.glob(ai_dir + 'cprCoronary/*.dcm'))
    vessel_id_map = json_load(ai_dir + 'map_coro_idx.txt')
    c2d_pts_dict = load_all_centerline(ai_dir + 'centerline/')
    ai_jd = json_load(ai_dir + 'diagnose_debug/diag_result.json')
    det_dir = ai_dir + 'diagnose_debug/detection/'
    mask_dir = ai_dir + 'diagnose_debug/segmentation/'
    ai_position_slice_dict = parse_ai_json(ai_jd)
    vessel_dcm_paths = defaultdict(list)
    mkdir_safe(save_dir)
    json_save(save_dir + 'ai_result.json', ai_jd)
    for dcm_path in dcm_paths:
        file_name = dcm_path.split('/')[-1][:-4]
        vessel_id, degree = file_name.split('_')
        vessel_dcm_paths[vessel_id].append([int(degree), file_name, dcm_path])
    for vessel_id in sorted(vessel_dcm_paths.keys()):
        position = vessel_id_map[vessel_id]
        for slice_id, (degree, file_name, dcm_path) in enumerate(sorted(vessel_dcm_paths[vessel_id])):
            img = gen_image_from_dcm(dcm_path, [(-200, 800)])[0]
            mask = load_image(dcm_path + '.bmp')
            position_slice = position + '_%d' % slice_id
            ai_nodes = ai_position_slice_dict[position_slice]
            c2d_pts = c2d_pts_dict[file_name]
            det_imgs = load_det_imgs(det_dir, file_name)
            masks = load_masks(mask_dir, file_name)
            draw_img = draw_image(img, mask, ai_nodes, c2d_pts, det_imgs, masks)
            cv2.imwrite(save_dir + '%s_%03d.jpg' % (position, degree), draw_img)
    ste_debug_dir = ai_dir + 'diagnose_debug/stenosis/'
    if os.path.exists(ste_debug_dir):
        if os.path.exists(save_dir + 'stenosis/'):
            shutil.rmtree(save_dir + 'stenosis/')
        shutil.copytree(ste_debug_dir, save_dir + 'stenosis/')
    return True


def collect(base_dir, save_dir):
    pids = get_pids(base_dir)
    for pid in pids:
        #if pid not in ['1004550']:
        #    continue
        cta_dirs = sorted(glob.glob('/data1/inputdata/%s/*/*_CTA/' % pid))
        for cta_dir in cta_dirs:
            dcm_paths = sorted(glob.glob(cta_dir[:-5] + '/*.dcm'))
            dcm = dicom_load(dcm_paths[0])
            sub_dir = dcm.PatientID + '_' + dcm.StudyInstanceUID + '_' + dcm.SeriesInstanceUID
            sub_save_dir = save_dir + sub_dir + '/'
            flag = draw_ai(cta_dir, sub_save_dir)
            print('%s: %s' % (sub_dir, 'OK' if flag else 'ERROR'))


def main():
    #base_dir = '/data2/zhangfd/data/cta/std_test/b1_150/'
    #base_dir = '/data2/zhangfd/data/cta/hospital/yanan0624/'
    base_dir = '/data2/zhangfd/data/cta/hospital/0703/jiangong/'
    save_dir = base_dir + 'ai_results/0630_final/'
    collect(base_dir, save_dir)


if __name__ == '__main__':
    main()

