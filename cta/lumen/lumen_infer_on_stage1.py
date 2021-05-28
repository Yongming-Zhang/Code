import cv2
import math
import glob
import numpy
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import SimpleITK as sitk


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


class LumenPatient(object):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def _load(self):
        pass


class LumenModel(object):
    def __init__(self, gpu_id, model_path):
        self.lumen_windows = [(300, 700), (-100, 700), (-500, 1000)]
        self.batch_size = 32
        self.patch_size_xy = 20
        self.patch_resize_xy = 40
        self.patch_size_z = 28
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.model = self._load_model()
        self.vessel_lumen_info = {}
        self.current_input_dir = ''

    def get_c3d_probs(self, patient, vessel_id, cl_range):
        if patient.input_dir != self.current_input_dir:
            self.clear()
            self.current_input_dir = patient.input_dir
        if vessel_id not in self.vessel_lumen_info:
            self.vessel_lumen_info[vessel_id] = {}
        lumen_dir = patient.input_dir + '_CTA/probe/%s/' % vessel_id
        if 'lumen_img_dict' not in self.vessel_lumen_info[vessel_id]:
            self.vessel_lumen_info[vessel_id]['lumen_img_dict'] = self._load_and_crop_lumen(lumen_dir)
        if 'c3d_lumen_indices' not in self.vessel_lumen_info[vessel_id]:
            spacing_x, spacing_y, spacing_z = patient.get_cta_pixel_spacings()
            c3d_pts = patient.centerline3d_dict[vessel_id]
            self.vessel_lumen_info[vessel_id]['c3d_lumen_indices'] = get_centerline3d_lumen_indices(
                c3d_pts, spacing_x, spacing_y, spacing_z)
        if 'lumen_probs_dict' not in self.vessel_lumen_info[vessel_id]:
            self.vessel_lumen_info[vessel_id]['lumen_probs_dict'] = {}
        c3d_i0, c3d_i1 = cl_range
        c3d_lumen_indices = self.vessel_lumen_info[vessel_id]['c3d_lumen_indices']
        lumen_indices = sorted(list(set([c3d_lumen_indices[i] for i in range(c3d_i0, c3d_i1 + 1)])))
        unpredicted_lumen_indices = []
        for lid in lumen_indices:
            if lid not in self.vessel_lumen_info[vessel_id]['lumen_probs_dict']:
                unpredicted_lumen_indices.append(lid)
        if len(unpredicted_lumen_indices) > 0:
            lumen_probs = self._lumen_predict(
                self.vessel_lumen_info[vessel_id]['lumen_img_dict'], unpredicted_lumen_indices)
            self.vessel_lumen_info[vessel_id]['lumen_probs_dict'].update(lumen_probs)
        lumen_probs_dict = self.vessel_lumen_info[vessel_id]['lumen_probs_dict']
        c3d_probs = {c3d_i: lumen_probs_dict[c3d_lumen_indices[c3d_i]] for c3d_i in range(c3d_i0, c3d_i1 + 1)}
        return c3d_probs

    def clear(self):
        self.vessel_lumen_info = {}

    def _load_model(self):
        model = LumenNet()
        model = model.cuda(self.gpu_id)
        model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        model.eval()
        return model

    def _gen_one_patch(self, lumen_index, lumen_img_dict):
        l0 = lumen_index - self.patch_size_z // 2
        l1 = l0 + self.patch_size_z
        empty_img = numpy.zeros([3, self.patch_resize_xy, self.patch_resize_xy], dtype='uint8')
        imgs = [lumen_img_dict.get(i, empty_img).reshape(
            1, 3, self.patch_resize_xy, self.patch_resize_xy) for i in range(l0, l1)]
        img_tensor = numpy.concatenate(imgs, axis=0)
        img_tensor = img_tensor.reshape(1, self.patch_size_z, 3, self.patch_resize_xy, self.patch_resize_xy)
        img_tensor = torch.Tensor(img_tensor).contiguous()
        img_tensor = img_tensor.float().div(255.)
        return img_tensor

    def _gen_batches(self, lumen_indices, lumen_img_dict):
        batches = []
        num_lumen = len(lumen_indices)
        for i in range(0, num_lumen, self.batch_size):
            batch_lumen_indices = lumen_indices[i: min(num_lumen, i + self.batch_size)]
            batch_patch_tensor = torch.cat([self._gen_one_patch(lid, lumen_img_dict) for lid in batch_lumen_indices], 0)
            batches.append(batch_patch_tensor)
        return batches

    def _load_and_crop_lumen(self, lumen_dir):
        lumen_paths = sorted(glob.glob(lumen_dir + '*.dcm'))
        lumen_img_dict = {}
        for lumen_path in lumen_paths:
            dcm = pydicom.dcmread(lumen_path)
            instance_num = int(dcm.InstanceNumber)
            imgs = gen_image_from_dcm(lumen_path, self.lumen_windows)
            rgb_img = numpy.zeros([self.patch_resize_xy, self.patch_resize_xy, 3], dtype='uint8')
            for c, img in enumerate(imgs):
                rgb_img[:, :, c] = cv2.resize(crop_img(img, self.patch_size_xy), (self.patch_resize_xy, self.patch_resize_xy))
            lumen_img_dict[instance_num] = rgb_img.transpose((2, 0, 1))
        return lumen_img_dict

    def _lumen_predict(self, lumen_img_dict, lumen_indices):
        if len(lumen_indices) == 0:
            return {}
        # print([(lid, self._gen_one_patch(lid, lumen_img_dict).mean()) for lid in lumen_indices])
        batches = self._gen_batches(lumen_indices, lumen_img_dict)
        all_probs = self._lumen_inference_batches(batches)
        lumen_probs = {lid: all_probs[i] for i, lid in enumerate(lumen_indices)}
        return lumen_probs

    def _lumen_inference_batches(self, batches):
        all_probs = []
        for batch_tensor in batches:
            batch_tensor = batch_tensor.cuda(self.gpu_id, async=True)
            if torch.__version__[:3] <= '0.3':
                batch_tensor = torch.autograd.Variable(batch_tensor, volatile=True)
            output = self.model(batch_tensor)
            probs = F.softmax(output, dim=1).data
            all_probs.append(probs)
        all_probs = torch.cat(all_probs).cpu().numpy()
        return all_probs
