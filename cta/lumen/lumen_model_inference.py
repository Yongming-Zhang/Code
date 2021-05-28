import sys
import cv2
import json
import math
import glob
import numpy
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple
import SimpleITK as sitk


# ********************************* Global Variables ********************************* #
Vessel = namedtuple('Vessel', 'vid c3d lids img')
C3dMapper = namedtuple('C3dMapper', 'pts3d vessel dist2_mat c3d_nearest_indices c3d_nearest_dist2s')
lumen_windows = [(300, 700), (-100, 700), (-500, 1000)]
batch_size, patch_size_xy, patch_resize_xy, patch_size_z = 32, 20, 40, 28
torch.cuda.manual_seed(1)
torch.cuda.set_device(int(sys.argv[1]))


# ********************************* Util ********************************* #
def gen_image_from_dcm(dcm_path, min_max_values):
    # print(dcm_path, type(dcm_path))
    img16_raw = numpy.float32(numpy.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(dcm_path)))))
    imgs = []
    for min_value, max_value in min_max_values:
        img16 = img16_raw.copy()
        min_value = img16.min() if min_value == 'min' else min_value
        max_value = img16.max() if max_value == 'max' else max_value
        img16[img16 > max_value] = max_value
        img16[img16 < min_value] = min_value
        img8u = numpy.uint8((img16 - min_value) * 255. / (max_value - min_value))
        imgs.append(img8u)
    return imgs


def crop_img(img, center_crop):
    img_h, img_w = img.shape
    x0, y0 = (img_w - center_crop) // 2, (img_h - center_crop) // 2
    img = img[y0:(y0 + center_crop), x0:(x0 + center_crop)]
    return img


def pairwise_dist2(x, y):
    n, m = x.shape[0], y.shape[0]
    x2 = (x**2).sum(axis=1).reshape(n, 1).repeat(m, axis=1)
    y2 = (y**2).sum(axis=1).reshape(1, m).repeat(n, axis=0)
    return x2 + y2 - 2 * numpy.dot(x, y.T)


# ********************************* Lumen model ********************************* #
def conv3d(in_planes, out_planes, kernel_size, stride, padding):
    module = nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True)
    )
    return module


class LumenNet(nn.Module):
    def __init__(self):
        super(LumenNet, self).__init__()
        feat_dim = 256
        num_classes = 5
        self.backbone = models.resnet18(pretrained=True)
        self.conv3d_1 = conv3d(128, feat_dim, (7, 1, 1), (2, 1, 1), (3, 0, 0))
        self.conv3d_2 = conv3d(feat_dim, feat_dim, (3, 1, 1), (2, 1, 1), (1, 0, 0))
        self.conv3d_3 = conv3d(feat_dim, feat_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv3d_4 = conv3d(feat_dim, feat_dim, (7, 1, 1), (1, 1, 1), (0, 0, 0))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)

    def _forward_backbone(self, x):
        b, d, c, h, w = x.size()
        x = x.view(b * d, c, h, w)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        bd, c, h, w = x.size()
        x = x.view(b, d, c, h, w).transpose(1, 2).contiguous()
        return x

    def forward(self, x):
        x = self._forward_backbone(x)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_lumen_model():
    model_path = '/breast_data/cta/new_data/b2_sg_407/model_data/lumen/lists/p32x32x32_w300-700_-100-700_-500-1000/'
    model_path += 'train_balance_label11111/models/resnet18_drop_l4c256/'
    model_path += 'ME12_B32_ft_rr_flip_crop28x20x20_resize40_lr0.01_epoch5.pkl'
    model = LumenNet()
    model = model.cuda()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    print('loaded %s' % model_path)
    return model


# ********************************* Lumen indices ********************************* #
def get_centerline3d_lumen_indices(c3d_pts, spacing_x, spacing_y, spacing_z, start_i=0):
    c3d_dist, lumen_i, lumen_indices = 0, start_i, [start_i] * len(c3d_pts)
    for i in range(1, len(c3d_pts)):
        (x0, y0, z0), (x1, y1, z1) = c3d_pts[i - 1], c3d_pts[i]
        c3d_dist += math.sqrt((spacing_x * (x1 - x0)) ** 2 + (spacing_y * (y1 - y0)) ** 2 + (spacing_z * (z1 - z0)) ** 2)
        if c3d_dist >= spacing_x:
            c3d_dist = 0
            lumen_i += 1
        lumen_indices[i] = lumen_i
    return lumen_indices


def get_cta_pixel_spacing(dcm_dir):
    spacing_x, spacing_y, zs = None, None, []
    for dcm_path in sorted(glob.glob(dcm_dir + '*.dcm')):
        dcm = pydicom.dcmread(dcm_path)
        spacing_x, spacing_y = float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1])
        zs.append(float(dcm.ImagePositionPatient[2]))
    spacing_z = abs(zs[-1] - zs[0]) / (len(zs) - 1)
    return spacing_x, spacing_y, spacing_z


# ********************************* Lumen images ********************************* #
def load_and_crop_lumen(lumen_dir):
    lumen_paths = sorted(glob.glob(lumen_dir + '*.dcm'))
    d = {}
    for lumen_path in lumen_paths:
        dcm = pydicom.dcmread(lumen_path)
        instance_num = int(dcm.InstanceNumber)
        imgs = gen_image_from_dcm(lumen_path, lumen_windows)
        rgb_img = numpy.zeros([patch_resize_xy, patch_resize_xy, 3], dtype='uint8')
        for c, img in enumerate(imgs):
            rgb_img[:, :, c] = cv2.resize(crop_img(img, patch_size_xy), (patch_resize_xy, patch_resize_xy))
        d[instance_num] = rgb_img.transpose((2, 0, 1))
    return d


def gen_one_patch(lumen_index, lumen_img_dict):
    l0 = lumen_index - patch_size_z // 2
    l1 = l0 + patch_size_z
    empty_img = numpy.zeros([3, patch_resize_xy, patch_resize_xy], dtype='uint8')
    imgs = [lumen_img_dict.get(i, empty_img).reshape(1, 3, patch_resize_xy, patch_resize_xy) for i in range(l0, l1)]
    img_tensor = numpy.concatenate(imgs, axis=0).reshape(1, patch_size_z, 3, patch_resize_xy, patch_resize_xy)
    img_tensor = torch.Tensor(img_tensor).contiguous()
    img_tensor = img_tensor.float().div(255.)
    return img_tensor


def gen_batches(lumen_indices, lumen_img_dict):
    batches = []
    num_lumen = len(lumen_indices)
    for i in range(0, num_lumen, batch_size):
        batch_lumen_indices = lumen_indices[i: min(num_lumen, i + batch_size)]
        batch_patch_tensor = torch.cat([gen_one_patch(lid, lumen_img_dict) for lid in batch_lumen_indices], 0)
        batches.append(batch_patch_tensor)
    return batches


# ********************************* Lumen prediction ********************************* #
def lumen_inference_batches(model, batches):
    all_probs = []
    for batch_tensor in batches:
        batch_tensor = batch_tensor.cuda(async=True)
        if torch.__version__[:3] <= '0.3':
            batch_tensor = torch.autograd.Variable(batch_tensor, volatile=True)
        output = model(batch_tensor)
        probs = F.softmax(output, dim=1).data
        all_probs.append(probs)
    all_probs = torch.cat(all_probs).cpu().numpy()
    return all_probs


def lumen_predict(model, lumen_img_dict, lumen_indices_dict):
    lumen_indices = sorted(list(set(lumen_indices_dict.values())))
    batches = gen_batches(lumen_indices, lumen_img_dict)
    all_probs = lumen_inference_batches(model, batches)
    lumen_probs = {lid: all_probs[i] for i, lid in enumerate(lumen_indices)}
    label_names = ['fp', 'low', 'cal', 'mix', 'ste']
    for cl_id, lid in sorted(lumen_indices_dict.items()):
        prob_str = '/'.join(['%s=%.4f' % (label_names[j], lumen_probs[lid][j]) for j in range(5)])
        pred = label_names[lumen_probs[lid].argmax()]
        print('cl=%d, lid=%d, pred=%s, probs: %s' % (cl_id, lid + 389, pred, prob_str))


# ********************************* Mapping predicted 3d points into lumen ********************************* #
def get_c3d_mapper(pts3d, vessel):
    d2_mat = pairwise_dist2(pts3d, vessel.c3d)
    c3d_nearest_indices = d2_mat.argmin(axis=1)
    c3d_nearest_dist2s = d2_mat.min(axis=1)
    mapper = C3dMapper(
        pts3d=pts3d, vessel=vessel,
        dist2_mat=d2_mat, c3d_nearest_indices=c3d_nearest_indices, c3d_nearest_dist2s=c3d_nearest_dist2s)
    return mapper


def find_best_vessel(c3d_mappers):
    min_dist, min_dist_mapper = 300, None
    for c3d_mapper in c3d_mappers:
        d = c3d_mapper.c3d_nearest_dist2s.max()
        if d < min_dist:
            min_dist = d
            min_dist_mapper = c3d_mapper
    return min_dist_mapper


def get_pred_lumen_indices(pred_pts, vessels):
    c3d_mappers = [get_c3d_mapper(pred_pts, vessel) for vessel in vessels]
    best_vessel_mapper = find_best_vessel(c3d_mappers)
    lumen_indices_dict = {}
    c3d_i0 = best_vessel_mapper.c3d_nearest_indices.min()
    c3d_i1 = best_vessel_mapper.c3d_nearest_indices.max()
    for i in range(c3d_i0, c3d_i1 + 1):
        li = best_vessel_mapper.vessel.lumen[i]
        lumen_indices_dict[i] = li
    return best_vessel_mapper.vessel, lumen_indices_dict


# ********************************* Main test ********************************* #
def proc_one_case(psid):
    root_dir = '/breast_data/cta/new_data/b2_sg_407/'
    base_dir = root_dir + 'raw/cpr_lumen_s9_n20/'
    centerline_dir = base_dir + psid + '/cpr/centerline/'
    lumen_dir = base_dir + psid + '/lumen/'
    cta_dir = root_dir + 'cta_dicom/' + psid + '/'
    spacing_x, spacing_y, spacing_z = get_cta_pixel_spacing(cta_dir)
    c3d_paths = sorted(glob.glob(centerline_dir + '*.3d'))
    vessels = []
    for c3d_path in c3d_paths:
        vid = c3d_path.split('_')[0]
        c3d_dict = json.load(open(c3d_path))
        c3d_pts = c3d_dict['points']
        lumen_indices = get_centerline3d_lumen_indices(c3d_pts, spacing_x, spacing_y, spacing_z)
        lumen_imgs = load_and_crop_lumen(lumen_dir + vid + '/')
        vessels.append(Vessel(vid=vid, c3d=c3d_pts, lids=lumen_indices, img=lumen_imgs))


def main():
    root_dir = '/breast_data/cta/new_data/b2_sg_407/'
    base_dir = root_dir + 'raw/cpr_lumen_s9_n20/'
    psid = '0784617_1.2.392.200036.9116.2.5.1.37.2417525831.1552459283.281312'
    vid = 'coro002'
    cl_range = [273, 510]
    centerline_dir = base_dir + psid + '/cpr/centerline/'
    lumen_dir = base_dir + psid + '/lumen/'
    cta_dir = root_dir + 'cta_dicom/' + psid + '/'
    spacing_x, spacing_y, spacing_z = get_cta_pixel_spacing(cta_dir)
    c3d_path = centerline_dir + vid + '.3d'
    c3d_dict = json.load(open(c3d_path))
    c3d_pts = c3d_dict['points']
    lumen_indices = get_centerline3d_lumen_indices(c3d_pts, spacing_x, spacing_y, spacing_z)
    lumen_img_dict = load_and_crop_lumen(lumen_dir + vid + '/')
    c3d_i0, c3d_i1 = cl_range
    pred_lumen_indices_dict = {i: lumen_indices[i] for i in range(c3d_i0, c3d_i1 + 1)}
    model = load_lumen_model()
    lumen_predict(model, lumen_img_dict, pred_lumen_indices_dict)


if __name__ == '__main__':
    main()

