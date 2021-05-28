import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
import nibabel as nib
import scipy.io as scio
import random
sys.path.insert(0, '../../')


def tensor_resize(image, shape, is_label=True):
    x = torch.from_numpy(image[np.newaxis, np.newaxis, ...])
    if is_label:
        y = F.interpolate(x.float(), shape, mode='nearest')
    else:
        y = F.interpolate(x.float(), shape, mode='trilinear', align_corners=True)
    y = y.numpy().astype(image.dtype)[0, 0]
    return y


def psf(pts, kernel=0, size=None, as_tuple=True):
    """
    point spread function
    Args:
        pts (list[float]): N points with K dim
        kenerl (int) : 0 = center, 1=8 points, 2=27 points
        size (img.size): K int
        as_tuple (bool): if output as tuple
    Return:
        pts_list:
            if as_tuple=True: (array, ) x K
            if as_tuple=False: N x K array

    Note: the kenerl points count is using 3-d input as reference.
    """
    if kernel == 1:
        pts = np.array(pts).astype(int)
    else:
        pts = np.array(pts).round().astype(int)
    if len(pts.shape) == 1:
        # dim -> 1 x dim
        pts = pts[None]
    if kernel > 0:
        dim = pts.shape[-1]
        if kernel == 1:
            neighbor_pts = np.stack(np.meshgrid(*[(0, 1)] * dim))
        elif kernel == 2:
            neighbor_pts = np.stack(np.meshgrid(*[(-1, 0, 1)] * dim))
        # N x dim x 1 + dim x 27 -> N x dim x 27
        pts = pts[..., None] + neighbor_pts.reshape(dim, -1)
        # N x dim x 27 -> N*27 x dim
        pts = pts.transpose(0, 2, 1).reshape(-1, dim)
        size = None if size is None else np.array(size) - 1
        pts = pts.clip(0, size)
    if as_tuple:
        pts = tuple(pts.T)
    return pts


def get_nii_data_path(base_dir, subject):
    nii_path = os.path.join(base_dir, '%s.nii' % subject)
    gz_path = os.path.join(base_dir, '%s.nii.gz' % subject)
    return nii_path if os.path.exists(nii_path) else gz_path


class VesselSegDataLoader(object):
    def __init__(self, subject, root_dir=None, img_dir=None, mask_dir=None, heatmap_dir=None, ske_dir=None, norm_z=0.5):
        if root_dir is None:
            root_dir = '/brain_data/dataset/ccta/'
        img_nii = nib.load(get_nii_data_path(root_dir + '/image/' if img_dir is None else img_dir, subject))
        image = img_nii.get_data()
        mask = nib.load(get_nii_data_path(root_dir + '/mask/' if mask_dir is None else mask_dir, subject))
        mask = mask.get_data()
        spacing = img_nii.header['pixdim'][1:4]
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        spacing = np.roll(spacing, 1, axis=0)
        heatmap = nib.load(get_nii_data_path(root_dir + '/heatmap/' if heatmap_dir is None else heatmap_dir, subject))
        heatmap = heatmap.get_data().astype('float32')
        heatmap = np.transpose(heatmap, (2, 0, 1))
        ske_dir = root_dir + '/skeleton/' if ske_dir is None else ske_dir
        points = scio.loadmat(ske_dir + '%s.mat' % subject)['points']
        points = np.roll(points, 1, axis=1)
        if img_nii.header['pixdim'][0] > 0:
            image = image[::-1, ...].copy()
            mask = mask[::-1, ...].copy()
            heatmap = heatmap[::-1, ...].copy()
            points = list(map(lambda x: np.array([mask.shape[0] - 1 - x[0], x[1], x[2]], 'int32'), points))
        if spacing[0] != norm_z and norm_z > 0:
            x, y, z = image.shape
            shape = (int(x * spacing[0] / norm_z), y, z)
            image = np.uint8(tensor_resize(image, shape, False))
            mask = np.uint8(tensor_resize(mask, shape, True))
            heatmap = np.float32(tensor_resize(heatmap, shape, True))
            points = list(map(lambda x: np.array([int(x[0] * spacing[0] / 0.5 + 0.5), x[1], x[2]], 'int32'), points))
        self.image_nii = img_nii
        self.image = image
        self.mask = mask
        self.heatmap = heatmap
        self.points = points
        self.subject = subject
        self.norm_z = norm_z

    def get_image_mask_heatmap_skepoints(self):
        return self.image_nii, self.image, self.mask, self.heatmap, self.points

    def get_skemask(self, bg_value=127, ske_value=255):
        skemask = np.ones_like(self.mask) * bg_value
        psf_points = psf(self.points, 2, self.mask.shape, as_tuple=True)
        skemask[psf_points] = self.mask[psf_points] * ske_value
        return skemask

    def save_data(self, img_path, mask_path, heatmap_path, ske_path):
        new_header = self.image_nii.header.copy()
        new_header['pixdim'][0] = -1
        scale_z = self.norm_z / new_header['pixdim'][3]
        new_header['pixdim'][3] = self.norm_z
        new_header['dim'][3] = int(new_header['dim'][3] / scale_z)
        new_header['qoffset_z'] *= scale_z
        new_header['srow_z'] = [z * scale_z for z in new_header['srow_z']]
        affine = self.image_nii.affine
        affine[2] *= scale_z
        image = np.transpose(self.image, (1, 2, 0))
        image_nii = nib.Nifti1Image(image, affine, header=new_header)
        nib.save(image_nii, img_path)
        mask = np.transpose(self.mask, (1, 2, 0))
        mask_nii = nib.Nifti1Image(mask, affine)
        nib.save(mask_nii, mask_path)
        heatmap = np.transpose(self.heatmap, (1, 2, 0))
        heatmap_nii = nib.Nifti1Image(heatmap, affine)
        nib.save(heatmap_nii, heatmap_path)
        points_dict = {'points': np.roll(self.points, 2, axis=1)}
        scio.savemat(ske_path, points_dict)
        # print(type(self.image), type(self.mask))
        return image_nii, mask_nii, heatmap_nii, points_dict


def load_vessel_seg_data(base_dir, pid):
    img_nii = nib.load(base_dir + 'image/' + pid + '.nii')
    image = img_nii.get_data()
    mask = nib.load(base_dir + 'mask/' + pid + '.nii').get_data()
    mask = mask.astype('bool').astype('uint8')
    spacing = img_nii.header['pixdim'][1:4]
    image = np.transpose(image, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))
    spacing = np.roll(spacing, 1, axis=0)
    heatmap = nib.load(base_dir + 'heatmap/' + pid + '.nii.gz').get_data().astype('float32')
    heatmap = np.transpose(heatmap, (2, 0, 1))
    points = scio.loadmat(base_dir + 'skeleton/%s.mat' % pid)['points']
    points = np.roll(points, 1, axis=1)
    if img_nii.header['pixdim'][0] > 0:
        image = image[::-1, ...].copy()
        mask = mask[::-1, ...].copy()
        heatmap = heatmap[::-1, ...].copy()
        points = list(map(lambda x: np.array([mask.shape[0]-1-x[0],x[1],x[2]], 'int32'), points))
    if spacing[0] != 0.5:
        x, y, z = image.shape
        shape = (int(x * spacing[0] / 0.5), y, z)
        image = tensor_resize(image, shape, False)
        mask = tensor_resize(mask, shape, True)
        heatmap = tensor_resize(heatmap, shape, True)
        points = list(map(lambda x: np.array([int(x[0] * spacing[0] / 0.5 + 0.5), x[1], x[2]], 'int32'), points))
    skemask = np.ones_like(mask) * 127
    psf_points = psf(points, 2, mask.shape, as_tuple=True)
    # print(len(points))
    skemask[psf_points] = mask[psf_points] * 255
    # for point in points:
    #    x, y, z = point
    #    skemask[x-1:x+2, y-1:y+2, z-1:z+2] = mask[x-1:x+2, y-1:y+2, z-1:z+2] * 255
    return image, mask, heatmap, skemask


def set_window_wl_ww(tensor, wl=225, ww=450):
    w_min, w_max = wl - ww // 2, wl + ww // 2
    tensor[tensor < w_min] = w_min
    tensor[tensor > w_max] = w_max
    tensor = ((1.0 * (tensor - w_min) / (w_max - w_min)) * 255).astype(np.uint8)
    return tensor


def sample_one_patch(img, mask, other_tensors, patch_size=(32, 384, 384)):
    vx, vy, vz = mask.shape
    px, py, pz = patch_size
    for i in range(100):
        x = random.randint(0, vx - px)
        y = random.randint(0, vy - py)
        z = random.randint(0, vz - pz)
        mask_patch = mask[x:x+px, y:y+py, z:z+pz]
        if mask_patch.sum() >= 10:
            break
    img_patch = img[x:x+px, y:y+py, z:z+pz]
    img_patch = set_window_wl_ww(img_patch, 400, 1000)
    flip_y = np.random.choice(2) * 2 - 1
    flip_z = np.random.choice(2) * 2 - 1
    img_patch = img_patch[..., ::flip_y, ::flip_z].copy()
    mask_patch = mask_patch[..., ::flip_y, ::flip_z].copy()
    other_patches = []
    for t in other_tensors:
        p = t[x:x+px, y:y+py, z:z+pz]
        other_patches.append(p[..., ::flip_y, ::flip_z].copy())
    param = [x, y, z, flip_y == -1, flip_z == -1]
    return img_patch, mask_patch, other_patches, param

