import torch
import torch.utils.data as data
import cv2
import numpy
import random


def parse_htiled_img(tiled_img, lumen_w):
    tile_h, tile_w = tiled_img.shape[:2]
    lumen_d = tile_w // lumen_w
    return [tiled_img[:, (z * lumen_w):(z * lumen_w + lumen_w), :] for z in range(lumen_d)]


def random_crop3d(imgs, random_crop):
    if random_crop is None:
        return imgs
    z, y, x = random_crop
    h, w, c = imgs[0].shape
    d = len(imgs)
    off_x = random.randint(0, w - x)
    off_y = random.randint(0, h - y)
    off_z = random.randint(0, d - z)
    new_imgs = [img[off_y:(off_y+y), off_x:(off_x+x)] for img in imgs[off_z:(off_z+z)]]
    return new_imgs


def center_crop3d(imgs, center_crop):
    if center_crop is None:
        return imgs
    z, y, x = center_crop
    h, w, c = imgs[0].shape
    d = len(imgs)
    off_x = (w - x) // 2
    off_y = (h - y) // 2
    off_z = (d - z) // 2
    new_imgs = [img[off_y:(off_y+y), off_x:(off_x+x)] for img in imgs[off_z:(off_z+z)]]
    return new_imgs


def random_rotate3d(imgs, random_rotate):
    if random_rotate is None:
        return imgs
    min_degree, max_degree = random_rotate
    rotate_degree = random.randint(min_degree, max_degree)
    h, w, c = imgs[0].shape
    new_imgs = [
        cv2.warpAffine(img, cv2.getRotationMatrix2D((w * 0.5, h * 0.5), rotate_degree, 1.0), (w, h)) for img in imgs]
    return new_imgs


def resize2d_list(imgs, resize):
    if resize is None:
        return imgs
    new_imgs = [cv2.resize(img, resize) for img in imgs]
    return new_imgs


def random_flip2d_img_list(imgs, random_flip2d):
    if random_flip2d is None or (not random_flip2d):
        return imgs
    flip_lr = random.randint(0, 1) * 2 - 1
    flip_tb = random.randint(0, 1) * 2 - 1
    new_imgs = [img[::flip_tb, ::flip_lr] for img in imgs]
    return new_imgs


class LumenListDataset(data.Dataset):
    def __init__(self, list_file_path, random_rotate, center_crop, random_crop, random_flip2d, resize2d, root_folder=''):
        # 1. Initialize file path or list of file names.
        self.path_size_labels = []
        self.random_rotate = random_rotate  # (0, 360)
        self.center_crop = center_crop      # (22, 22, 22)
        self.random_crop = random_crop      # (18, 18, 18)
        self.random_flip2d = random_flip2d  # True
        self.resize = resize2d              # (56, 56)
        self.root_folder = root_folder
        with open(list_file_path) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    path, size_str, label = line.split(' ')
                    size = [int(x) for x in size_str.split('x')]
                    self.path_size_labels.append([path, size, int(label)])
        print('loaded %d items from %s' % (len(self.path_size_labels), list_file_path))

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        path, size, label = self.path_size_labels[index]
        d, h, w = size
        tiled_img = cv2.imread(self.root_folder + path)
        imgs = [tiled_img[:, (z*w):(z*w+w), :] for z in range(d)]
        imgs = random_rotate3d(imgs, self.random_rotate)
        imgs = center_crop3d(imgs, self.center_crop)
        imgs = random_crop3d(imgs, self.random_crop)
        imgs = random_flip2d_img_list(imgs, self.random_flip2d)
        imgs = resize2d_list(imgs, self.resize)
        h, w, c = imgs[0].shape
        img_tensor = numpy.concatenate([img.reshape(1, h, w, c) for img in imgs], axis=0)
        img_tensor = img_tensor.transpose((0, 3, 1, 2))
        img_tensor = torch.Tensor(img_tensor).contiguous()
        img_tensor = img_tensor.float().div(255.)
        return img_tensor, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.path_size_labels)


class LumenListWholeImageDataset(data.Dataset):
    def __init__(self, list_file_path, image_dir, mask_dir, lumen_patch_size,
                 sample_positive, random_rotate, center_crop, random_crop, random_flip2d, resize2d):
        # 1. Initialize file path or list of file names.
        self.paths = []
        self.sample_positive = sample_positive  # positive probability
        self.random_rotate = random_rotate      # (0, 360)
        self.center_crop = center_crop          # (256, 40, 40)
        self.random_crop = random_crop          # (256, 32, 32)
        self.random_flip2d = random_flip2d      # True
        self.resize = resize2d                  # (64, 64)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.lumen_patch_size = lumen_patch_size
        with open(list_file_path) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    self.paths.append(line)
        self.paths *= 5
        print('loaded %d items from %s' % (len(self.paths), list_file_path))

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Pre-process the data (e.g. torchvision.Transform).
        #   1) sample one index
        #   2) transform: random crop_z, random_rotate, center_crop, random crop x/y, random_flip2d, resize2d
        # 3. Return a data pair (e.g. image and label).
        path = self.paths[index] + '.png'
        lumen_h, lumen_w = self.lumen_patch_size
        tiled_img = cv2.imread(self.image_dir + path)
        imgs = parse_htiled_img(tiled_img, lumen_w)
        mask = cv2.imread(self.mask_dir + path, cv2.IMREAD_GRAYSCALE)
        assert len(imgs) == mask.shape[0], 'Sizes of lumen image and mask are not match: %d vs %d' % (
            len(imgs), mask.shape[0])
        # process
        imgs, mask = self._crop_patch(imgs, mask)
        imgs = random_rotate3d(imgs, self.random_rotate)
        imgs = center_crop3d(imgs, self.center_crop)
        imgs = random_crop3d(imgs, self.random_crop)
        imgs = random_flip2d_img_list(imgs, self.random_flip2d)
        imgs = resize2d_list(imgs, self.resize)
        h, w, c = imgs[0].shape
        img_tensor = numpy.concatenate([img.reshape(1, h, w, c) for img in imgs], axis=0)
        img_tensor = img_tensor.transpose((0, 3, 1, 2))
        img_tensor = torch.Tensor(img_tensor).contiguous()
        img_tensor = img_tensor.float().div(255.)
        msk_tensor = torch.Tensor(mask).contiguous()
        return img_tensor, msk_tensor

    def _crop_patch(self, imgs, mask):
        imgs, mask = self._pad_img_msk_z(imgs, mask)
        crop_d, crop_h, crop_w = self.center_crop
        lumen_d = len(imgs)
        mask_bin = numpy.int32(mask > 0)
        pos_num = numpy.sum(mask_bin)
        if self.sample_positive <= 0 or pos_num == 0:
            off_z = random.randint(0, lumen_d - crop_d)
        else:
            pos_inter_list = [0] * lumen_d
            pos_inter_list[0] = mask_bin[0, 0]
            for z in range(1, lumen_d):
                pos_inter_list[z] = pos_inter_list[z - 1] + mask_bin[z, 0]
            pos_zs = []
            for z in range(0, lumen_d - crop_d + 1):
                pos_num_crop_d = pos_inter_list[z + crop_d - 1] - pos_inter_list[z]
                if pos_num_crop_d > 0:
                    pos_zs += [z] * pos_num_crop_d
            if len(pos_zs) == 0:
                print(mask_bin[:, 0].nonzero())
                print(pos_num)
                print(pos_inter_list)
                print()
            if random.random() <= self.sample_positive:
                off_z = random.choice(pos_zs)
            else:
                off_z = random.randint(0, lumen_d - crop_d)
        imgs = imgs[off_z: (off_z + crop_d)]
        mask = mask[off_z: (off_z + crop_d), 0]
        return imgs, mask

    def _pad_img_msk_z(self, imgs, mask):
        crop_d, crop_h, crop_w = self.center_crop
        lumen_d = len(imgs)
        if lumen_d >= crop_d:
            return imgs, mask
        pad_z0 = (crop_d - lumen_d) // 2
        pad_z1 = crop_d - lumen_d - pad_z0
        h, w, c = imgs[0].shape
        empty_img = numpy.zeros([h, w, 3], dtype='uint8')
        imgs = [empty_img] * pad_z0 + imgs + [empty_img] * pad_z1
        mask = numpy.vstack([numpy.zeros([pad_z0, 1], dtype='uint8'), mask, numpy.zeros([pad_z1, 1], dtype='uint8')])
        return imgs, mask

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.paths)


