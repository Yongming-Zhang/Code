from lumen_util import *
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import shutil
import numpy
import logging
import logging.config
from collections import defaultdict
from lumen_dataset import LumenListWholeImageDataset, parse_htiled_img, center_crop3d, resize2d_list
from lib.models.util import adjust_learning_rate
from lib.utils.mio import create_log_config, mkdir_safe, load_string_list, save_string_list, save_image
from lib.utils.meter import AverageMeter, AUCMeter, recall_precision, accuracy, confusion_matrix
from lib.image.draw import draw_texts, tile_images
import time


feat_dim, num_conv_per_stage = 256, 3
# seg_net_name = 'unet_psp'
# seg_net_name = 'simple'
gpu_id = int(sys.argv[1])
optimizer_name = sys.argv[2]
base_lr, snapshot = float(sys.argv[3]), 1
seg_net_name = sys.argv[4]
pos_weight = int(sys.argv[5])
backbone = 'resnet18_l4c%d' % feat_dim
base_dir = '/breast_data/cta/new_data/b2_sg_407/model_data/lumen_v2/layer_seg/'
list_dir = base_dir + 'lists/'
image_dir, mask_dir = base_dir + 'image_win1_p60/', base_dir + 'layer_mask/'
train_list_path = list_dir + 'train_img_list.txt'
valid_list_path = list_dir + 'valid_img_list.txt'
train_batch_size = 16
# optimizer_name = 'Adam'
num_classes = 5
is_pretrained = True
center_crop_z, center_crop_xy, random_crop_xy, resize_xy = 128, 32, 28, 56
lumen_patch_size = (60, 60)
sample_positive_prob = 0.9
torch.cuda.manual_seed(1)
torch.cuda.set_device(gpu_id)
label_dict = ['neg', 'low', 'cal', 'mix', 'ste']


def conv3d(in_planes, out_planes, kernel_size, stride, padding):
    module = nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True)
    )
    return module


def build_down_trans_up_stage(input_dim, output_dim, depth_dim, num_conv_layers):
    down_convs = []
    for i in range(num_conv_layers):
        if i == 0:
            k, s, p = (3, 3, 3), (2, 1, 1), (1, 1, 1)
            layer_input_dim = input_dim
        else:
            k, s, p = (3, 1, 1), (1, 1, 1), (1, 0, 0)
            layer_input_dim = output_dim
        down_convs.append(conv3d(layer_input_dim, output_dim, k, s, p))
    down = nn.Sequential(*down_convs)
    trans = nn.AdaptiveAvgPool3d((depth_dim // 2, 1, 1))
    fusion = conv3d(output_dim * 2, output_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    up = nn.AdaptiveAvgPool3d((depth_dim, 1, 1))
    return down, trans, fusion, up


class LumenLayerSegNet(nn.Module):
    def __init__(self):
        """            [trans0]                                                                                        [fusion0]
        x ----------------------------------------------------------------------------------------------> x_t0 + x_up1 -------> x_f0
         \ [down1]     [trans1]                                                                         [fusion1]   / [up1]
         x_d1/2 ---------------------------------------------------------------------> x_t1/2 + x_up2/2 -------> x_f1/2
          \ [down2]    [trans2]                                                        [fusion2]   / [up2]
          x_d2/4 ---------------------------------------------------> x_t2/4 + x_up3/4 -------> x_f2/4
           \ [down3]   [trans3]                                       [fusion3]   / [up3]
           x_d3/8 ---------------------------------> x_t3/8 + x_up4/8 -------> x_f3/8
            \ [down4]  [trans4]                      [fusion4]   / [up4]
            x_d4/16 ------------> x_t4/16 + x_up5/16 -------> x_f4/16
             \ [psp]  [trans_psp]            / [psp_up]
             x_d5_psp ----------> x_t5_psp/1x1x1
        """
        super(LumenLayerSegNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone_feat_dim = 256 if 112 >= resize_xy > 56 else 128
        self.conv0 = conv3d(self.backbone_feat_dim, feat_dim, (7, 3, 3), (1, 1, 1), (3, 1, 1))
        self.trans0 = nn.AdaptiveAvgPool3d((center_crop_z, 1, 1))
        self.fusion0 = conv3d(feat_dim * 2, feat_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0))

        input_depth_dim = center_crop_z
        self.down1, self.trans1, self.fusion1, self.up1 = build_down_trans_up_stage(
            feat_dim, feat_dim, input_depth_dim, num_conv_per_stage)
        input_depth_dim = center_crop_z // 2
        self.down2, self.trans2, self.fusion2, self.up2 = build_down_trans_up_stage(
            feat_dim, feat_dim, input_depth_dim, num_conv_per_stage)
        input_depth_dim = center_crop_z // 4
        self.down3, self.trans3, self.fusion3, self.up3 = build_down_trans_up_stage(
            feat_dim, feat_dim, input_depth_dim, num_conv_per_stage)
        input_depth_dim = center_crop_z // 8
        self.down4, self.trans4, self.fusion4, self.up4 = build_down_trans_up_stage(
            feat_dim, feat_dim, input_depth_dim, num_conv_per_stage)

        input_depth_dim = center_crop_z // 16
        self.psp_down = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.psp_trans = conv3d(feat_dim, feat_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0))
        self.psp_up = nn.AdaptiveAvgPool3d((input_depth_dim, 1, 1))

        self.fc = nn.Conv3d(feat_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_backbone(self, x):
        b, d, c, h, w = x.size()
        x = x.view(b * d, c, h, w)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # print('layer0:', x.size())
        x = self.backbone.layer1(x)
        # print('layer1:', x.size())
        x = self.backbone.layer2(x)
        if self.backbone_feat_dim == 256:
            x = self.backbone.layer3(x)
        # print('layer2:', x.size())
        bd, c, h, w = x.size()
        x = x.view(b, d, c, h, w).transpose(1, 2).contiguous()
        return x

    def forward(self, x):
        x = self._forward_backbone(x)
        x = self.conv0(x)
        b, c, d, h, w = x.size()
        # print('backbone:', x.size())
        # downsample
        x_d1_ds2 = self.down1(x)
        x_d2_ds4 = self.down2(x_d1_ds2)
        x_d3_ds8 = self.down3(x_d2_ds4)
        x_d4_ds16 = self.down4(x_d3_ds8)
        # psp
        x_d5_psp = self.psp_down(x_d4_ds16)
        x_t5_psp = self.psp_trans(x_d5_psp)
        # upsample and fusion
        x_up5_ds16 = self.psp_up(x_t5_psp)
        x_up4_ds8 = self.up4(self.fusion4(torch.cat([self.trans4(x_d4_ds16), x_up5_ds16], dim=1)))
        x_up3_ds4 = self.up3(self.fusion3(torch.cat([self.trans3(x_d3_ds8), x_up4_ds8], dim=1)))
        x_up2_ds2 = self.up2(self.fusion2(torch.cat([self.trans2(x_d2_ds4), x_up3_ds4], dim=1)))
        x_up1 = self.up1(self.fusion1(torch.cat([self.trans1(x_d1_ds2), x_up2_ds2], dim=1)))
        x_f0 = self.fusion0(torch.cat([self.trans0(x), x_up1], dim=1))
        x_softmax = self.fc(x_f0).view(b, num_classes, d)
        return x_softmax


class LumenSimpleLayerSegNet(nn.Module):
    def __init__(self):
        super(LumenSimpleLayerSegNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone_feat_dim = 256 if 112 >= resize_xy > 56 else 128
        self.conv0 = conv3d(self.backbone_feat_dim, feat_dim, (7, 3, 3), (1, 1, 1), (3, 1, 1))
        self.conv1s = nn.Sequential(
            conv3d(feat_dim, feat_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            conv3d(feat_dim, feat_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            # conv3d(feat_dim, feat_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            # conv3d(feat_dim, feat_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )
        self.pool = nn.AdaptiveAvgPool3d((center_crop_z, 1, 1))

        self.fc = nn.Conv3d(feat_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_backbone(self, x):
        b, d, c, h, w = x.size()
        x = x.view(b * d, c, h, w)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # print('layer0:', x.size())
        x = self.backbone.layer1(x)
        # print('layer1:', x.size())
        x = self.backbone.layer2(x)
        if self.backbone_feat_dim == 256:
            x = self.backbone.layer3(x)
        # print('layer2:', x.size())
        bd, c, h, w = x.size()
        x = x.view(b, d, c, h, w).transpose(1, 2).contiguous()
        return x

    def forward(self, x):
        x = self._forward_backbone(x)
        b, c, d, h, w = x.size()
        x = self.conv0(x)
        x = self.conv1s(x)
        x = self.pool(x)
        x_softmax = self.fc(x).view(b, num_classes, d)
        return x_softmax


def inference_one_image(model, htiled_img):
    imgs = parse_htiled_img(htiled_img, lumen_patch_size[0])
    h, w, c = imgs[0].shape
    # pad image
    crop_d, pad_z0 = center_crop_z, 0
    raw_lumen_d = len(imgs)
    if raw_lumen_d < crop_d:
        pad_z0 = (crop_d - raw_lumen_d) // 2
        pad_z1 = crop_d - raw_lumen_d - pad_z0
        empty_img = numpy.zeros([h, w, 3], dtype='uint8')
        imgs = [empty_img] * pad_z0 + imgs + [empty_img] * pad_z1
    # forward sliding window
    lumen_d = len(imgs)
    stride = crop_d // 2
    sum_probs = numpy.zeros([lumen_d, num_classes], dtype='float32')
    counts = numpy.zeros([lumen_d, num_classes], dtype='float32')
    for z in range(0, lumen_d, stride):
        z0 = min(lumen_d - crop_d, z)
        z1 = z0 + crop_d
        sub_imgs = center_crop3d(imgs[z0:z1], [center_crop_z, random_crop_xy, random_crop_xy])
        sub_imgs = resize2d_list(sub_imgs, (resize_xy, resize_xy))
        img_tensor = numpy.concatenate([img.reshape(1, resize_xy, resize_xy, c) for img in sub_imgs], axis=0)
        img_tensor = img_tensor.transpose((0, 3, 1, 2))
        img_tensor = torch.Tensor(img_tensor).contiguous()
        img_tensor = img_tensor.float().div(255.)[None].cuda()
        softmax = model(img_tensor)
        probs = F.softmax(softmax, dim=1).data.transpose(1, 2).contiguous().cpu().numpy().reshape(crop_d, num_classes)
        sum_probs[z0:z1, :] += probs
        counts[z0:z1, :] += 1
        if z + crop_d >= lumen_d:
            break
    probs = sum_probs / counts
    return probs[pad_z0: (pad_z0 + raw_lumen_d)]


def main(model_name, lr_step_values, max_epoch):
    model_save_dir = train_list_path[:-4] + '/models/%s_%s3/' % (model_name, seg_net_name)
    size_str = 'crop%dx%dx%d_resize%d_sp%s_%s_pw%d' % (
        center_crop_z, random_crop_xy, random_crop_xy, resize_xy, str(sample_positive_prob), optimizer_name, pos_weight)
    model_save_prefix = model_save_dir + 'ME%d_B%d%s_rr_flip_%s_lr%s' % (
        max_epoch, train_batch_size, '_ft' if is_pretrained else '', size_str, str(base_lr))
    logging.config.fileConfig(create_log_config(model_save_prefix + '.log', 'logging.conf'))
    logger = logging.getLogger('finger')
    logger.info('Name: ' + model_name)
    logger.info('GPU_id: %d' % gpu_id)
    logger.info('Base_lr: %s' % str(base_lr))
    logger.info('Nesterov: True')
    logger.info('Max_epoch: %d' % max_epoch)
    logger.info('Step_values: %s' % str(lr_step_values))
    logger.info('Model_dir: %s' % os.path.abspath(model_save_dir))
    logger.info('Model_prefix: %s' % model_save_prefix)

    center_crop_train = (center_crop_z, center_crop_xy, center_crop_xy)
    random_crop_train = (center_crop_z, random_crop_xy, random_crop_xy)
    resize2d = (resize_xy, resize_xy)
    train_dataset = LumenListWholeImageDataset(
        train_list_path, image_dir, mask_dir, lumen_patch_size, sample_positive_prob,
        (0, 360), center_crop_train, random_crop_train, True, resize2d)
    logger.info('Train_list: ' + train_list_path)
    image, label = train_dataset[0]
    print(image.size(), label.size(), label[:10])

    logger.info('Test_list: ' + valid_list_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    model = LumenSimpleLayerSegNet() if seg_net_name == 'simple' else LumenLayerSegNet()
    model = model.cuda()
    params = model.parameters()
    logger.info('Parameters: {}'.format(sum([p.data.nelement() for p in params])))
    ce_weights = torch.FloatTensor([1] + [pos_weight] * (num_classes - 1))
    criterion = nn.CrossEntropyLoss(weight=ce_weights).cuda()
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), base_lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), base_lr, weight_decay=0.0001)

    # test(valid_list_path, model, logger)
    for epoch in range(max_epoch):
        cur_lr = adjust_learning_rate(base_lr, lr_step_values, optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, cur_lr, logger)
        test(valid_list_path, model, logger)
        if (epoch + 1) % snapshot == 0 or epoch == (max_epoch - 1):
            state = model.state_dict()
            save_path = model_save_prefix + '_epoch%d.pkl' % (epoch + 1)
            torch.save(state, save_path)
            logger.info('Snapshot: ' + save_path)


def draw_and_save_data(images, target, save_path):
    images = numpy.uint8(images.cpu().numpy() * 255)
    masks = numpy.uint8(target.cpu().numpy())
    batch_size, batch_image_depth, lumen_c, lumen_h, lumen_w = images.shape
    draw_rows = []
    img_head_h = 20
    for i in range(batch_size):
        img = images[i].transpose((0, 2, 3, 1))
        msk = masks[i]
        draw_cols = []
        for z in range(batch_image_depth):
            draw_img = numpy.vstack([numpy.zeros([img_head_h, lumen_w, 3], dtype='uint8'), img[z]])
            label_str = '  %d' % msk[z]
            draw_texts(draw_img, (0, img_head_h + 1), [label_str], (255, 0, 0), is_bold=False, thick=1, font_scale=0.4)
            draw_cols.append(draw_img)
        draw_rows.append(numpy.hstack(draw_cols))
    canvas = numpy.vstack(draw_rows)
    save_image(save_path, canvas)


def train(loader, model, criterion, optimizer, epoch, epoch_lr, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    recalls = [AverageMeter() for i in range(num_classes)]
    precisions = [AverageMeter() for i in range(num_classes)]
    model.train()
    end = time.time()
    all_probs, all_labels = [], []
    # debug_dir = train_list_path[:-4] + '/debug/train_imgs/'
    # mkdir_safe(debug_dir)
    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.long().cuda(async=True)
        images = images.cuda()
        # draw_and_save_data(images, target, debug_dir + '%04d.png' % i)
        batch_size = images.size(0)
        batch_image_depth = images.size(1)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        probs = F.softmax(output, dim=1).data.transpose(1, 2).contiguous()
        probs = probs.view(batch_size * batch_image_depth, num_classes)
        target = target.view(batch_size * batch_image_depth)
        all_probs.append(probs)
        all_labels.append(target.view(batch_size * batch_image_depth, 1))
        # measure accuracy and record loss
        acc1 = accuracy(probs, target, topk=(1,))[0]
        accuracies.update(acc1 / 100., batch_size)
        losses.update(loss.item() if torch.__version__[:3] >= '0.4' else loss.data[0], batch_size)
        for cid in range(num_classes):
            correct_count, gt_count, pred_count = recall_precision(probs, target, cid)
            recalls[cid].update(correct_count / (gt_count + 1e-5), gt_count)
            precisions[cid].update(correct_count / (pred_count + 1e-5), pred_count)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0:
            line = 'Epoch: %d(%d/%d), lr=%.6f, ' % (epoch, i + 1, len(loader), epoch_lr)
            line += 'Time=%.2fs, Data=%.2fs, Loss=%.6f, ' % (batch_time.sum, data_time.sum, losses.avg)
            line += 'Accuracy=%.2f%%, Details:' % (accuracies.avg * 100.)
            logger.info(line)
            cm = confusion_matrix(torch.cat(all_probs), torch.cat(all_labels))
            for cid in range(num_classes):
                recall = recalls[cid].avg * 100.0
                pos_count = recalls[cid].count
                prec = precisions[cid].avg * 100.0
                msg = '    class_%d: recall=%.2f%%, precision=%.2f%%, pos_count=%d, ' % (cid, recall, prec, pos_count)
                msg += ', '.join(['pred_%d=%03d' % (cj, cm[cid, cj]) for cj in range(num_classes)])
                logger.info(msg)
            all_probs = []
            all_labels = []
            batch_time.reset()
            data_time.reset()
            losses.reset()
            accuracies.reset()
            for cid in range(num_classes):
                recalls[cid].reset()
                precisions[cid].reset()


def test(img_list_path, model, logger):
    batch_time = AverageMeter()
    accuracies = AverageMeter()
    recalls = [AverageMeter() for i in range(num_classes)]
    bin_recalls = [AverageMeter() for i in range(num_classes)]
    precisions = [AverageMeter() for i in range(num_classes)]
    bin_auc_meter = AUCMeter()
    all_probs, all_labels = [], []
    bin_correct_num, bin_total_num, bin_correct_pos_num, bin_total_pos_num, bin_pred_pos_num = 0, 0, 0, 0, 0
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for img_path in load_string_list(img_list_path):
        htiled_img = cv2.imread(image_dir + img_path + '.png')
        probs = inference_one_image(model, htiled_img)
        batch_size = probs.shape[0]
        mask = cv2.imread(mask_dir + img_path + '.png', cv2.IMREAD_GRAYSCALE).reshape((batch_size,))
        probs_tensor, mask_tensor = torch.Tensor(probs), torch.LongTensor(mask)
        all_probs.append(probs_tensor)
        all_labels.append(mask_tensor.view(batch_size, 1))
        # measure accuracy and record loss
        acc1 = accuracy(probs_tensor, mask_tensor, topk=(1,))[0]
        accuracies.update(acc1 / 100., batch_size)
        pos_scores = 1 - probs[:, 0]
        bin_labels = numpy.int32(mask > 0)
        bin_preds = numpy.int32(probs.argmax(axis=1) > 0)
        for cid in range(num_classes):
            correct_count, gt_count, pred_count = recall_precision(probs_tensor, mask_tensor, cid)
            correct_count_bin = numpy.sum((mask == cid) * (bin_preds == (cid > 0)))
            recalls[cid].update(correct_count / (gt_count + 1e-5), gt_count)
            bin_recalls[cid].update(correct_count_bin / (gt_count + 1e-5), gt_count)
            precisions[cid].update(correct_count / (pred_count + 1e-5), pred_count)
        bin_correct_num += numpy.sum(bin_labels == bin_preds)
        bin_total_num += batch_size
        bin_correct_pos_num += numpy.sum(bin_labels * bin_preds)
        bin_total_pos_num += numpy.sum(bin_labels)
        bin_pred_pos_num += numpy.sum(bin_preds)
        # print(
        #   acc1, batch_size, bin_preds.shape, numpy.sum(bin_preds),
        #   bin_labels.shape, numpy.sum(bin_labels), numpy.sum(bin_labels * bin_preds))
        bin_auc_meter.update(pos_scores, mask_tensor > 0)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    line = 'Test: Time=%.3fs/batch, Accuracy=%.2f%%, Details:' % (batch_time.avg, accuracies.avg * 100.)
    logger.info(line)
    line = '    Binary: Accuracy=%.2f%%(%d/%d), AUC=%.2f%%. Positive: recall=%.2f%%(%d/%d), precision=%.2f%%(%d/%d)' % (
        bin_correct_num * 100. / bin_total_num, bin_correct_num, bin_total_num,
        bin_auc_meter.get_auc(),
        bin_correct_pos_num * 100. / bin_total_pos_num, bin_correct_pos_num, bin_total_pos_num,
        bin_correct_pos_num * 100. / bin_pred_pos_num, bin_correct_pos_num, bin_pred_pos_num,
    )
    logger.info(line)
    cm = confusion_matrix(torch.cat(all_probs), torch.cat(all_labels))
    for cid in range(num_classes):
        recall = recalls[cid].avg * 100.0
        recall_bin = bin_recalls[cid].avg * 100.0
        pos_count = recalls[cid].count
        prec = precisions[cid].avg * 100.0
        msg = '    Class_%d: recall=%.2f%%, bin_recall=%.2f%%, precision=%.2f%%, pos_count=%06d, ' % (
            cid, recall, recall_bin, prec, pos_count)
        msg += ', '.join(['pred_%d=%06d' % (cj, cm[cid, cj]) for cj in range(num_classes)])
        logger.info(msg)


def get_rate_str(find_num, total_num):
    rate = find_num * 100. / (total_num + 1e-5)
    return '%.2f%%(%d/%d)' % (rate, find_num, total_num)


def draw_one_case(htiled_img, labels, probs, save_path):
    imgs = parse_htiled_img(htiled_img, lumen_patch_size[0])
    img_head_h, num_per_row, unit_size = 18, int(round(math.sqrt(len(imgs) * 2))), 112
    imgs = resize2d_list(imgs, (unit_size, unit_size))
    draw_imgs = []
    for i in range(len(imgs)):
        img_head = numpy.ones([img_head_h, unit_size, 3], dtype='uint8')
        prediction = probs[i].argmax()
        label = int(labels[i])
        pred_prob, label_prob = probs[i, prediction], probs[i, label]
        if label == prediction:
            s = '%d(%.2f)' % (label, pred_prob)
            color = (0, 255, 0)
            img_head *= 0
        else:
            s = '%d/%d(%.2f/%.2f)' % (label, prediction, label_prob, pred_prob)
            color = (0, 0, 255)
            img_head *= 255
        draw_img = numpy.vstack([img_head, imgs[i]])
        draw_texts(draw_img, (0, img_head_h + 1), [s], color, is_bold=False, thick=1, font_scale=0.4)
        draw_imgs.append(draw_img)
    canvas = tile_images(draw_imgs, num_per_row)
    save_image(save_path, canvas)


class BinaryPredictionCount(object):
    def __init__(self, num_cls=1):
        self.num_cls = num_cls
        self.pred_num = numpy.zeros([num_cls], dtype='int32')
        self.find_num = numpy.zeros([num_cls], dtype='int32')
        self.gt_num = numpy.zeros([num_cls], dtype='int32')

    def append(self, preds, labels):
        bin_preds = preds > 0
        cur_pred_num = numpy.zeros([self.num_cls], dtype='int32')
        cur_find_num = numpy.zeros([self.num_cls], dtype='int32')
        cur_gt_num = numpy.zeros([self.num_cls], dtype='int32')
        for i in range(self.num_cls):
            cur_pred_num[i] = numpy.sum(preds == i)
            label_i_mask = labels == i
            cur_gt_num[i] = numpy.sum(label_i_mask)
            cur_find_num[i] = numpy.sum((bin_preds if i > 0 else (1 - bin_preds)) * label_i_mask)
        self.pred_num += cur_pred_num
        self.find_num += cur_find_num
        self.gt_num += cur_gt_num

    def verbose(self):
        recall_strs, prec_strs = [], []
        for i in range(self.num_cls):
            recall_strs.append('%s=%s' % (label_dict[i], get_rate_str(self.find_num[i], self.gt_num[i])))
            prec_strs.append('%s=%s' % (label_dict[i], get_rate_str(self.find_num[i], self.pred_num[i])))
        recall_str = 'Recall: %s' % (', '.join(recall_strs))
        prec_str = 'Precision: %s' % (', '.join(prec_strs))
        return recall_str, prec_str


def generate_bad_cases(model_name):
    random_crop_z_, center_crop_xy_, random_crop_xy_, resize_xy_ = 128, 32, 28, 56
    sample_pos_prob_, pos_weight_ = 0.9, 10
    optimizer_name_ = 'Adam'
    max_epoch, model_epoch = 32, 32
    set_name = 'valid_new'

    model_save_dir = train_list_path[:-4] + '/models/%s_%s3/' % (model_name, seg_net_name)
    size_str = 'crop%dx%dx%d_resize%d_sp%s_%s_pw%d' % (
        random_crop_z_, random_crop_xy_, random_crop_xy_, resize_xy_, str(sample_pos_prob_), optimizer_name_, pos_weight_)
    model_save_prefix = model_save_dir + 'ME%d_B%d%s_rr_flip_%s_lr%s' % (
        max_epoch, train_batch_size, '_ft' if is_pretrained else '', size_str, str(base_lr))
    model_path = model_save_prefix + '_epoch%d.pkl' % model_epoch
    base_save_dir = model_path[:-4] + '/'
    save_dir = base_save_dir + '%s_bad/' % set_name
    bin_error_save_dir = save_dir + 'bin_error/'
    mkdir_safe(bin_error_save_dir)
    image_list_path = valid_list_path if set_name == 'valid_new' else train_list_path

    model = LumenSimpleLayerSegNet() if seg_net_name == 'simple' else LumenLayerSegNet()
    model = model.cuda()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print(model_path)
    model.eval()
    bin_pred, bin_pred_v2 = BinaryPredictionCount(5), BinaryPredictionCount(5)
    for img_path in load_string_list(image_list_path):
        htiled_img = cv2.imread(image_dir + img_path + '.png')
        probs = inference_one_image(model, htiled_img)
        batch_size = probs.shape[0]
        mask = cv2.imread(mask_dir + img_path + '.png', cv2.IMREAD_GRAYSCALE).reshape((batch_size,))
        pred = probs.argmax(axis=1)
        pred_v2 = numpy.int32(probs[:, 0] >= 0.5) * (probs[:, 1:].argmax(axis=1) + 1)
        bin_pred.append(pred, mask)
        bin_pred_v2.append(pred_v2, mask)
        draw_one_case(htiled_img, mask, probs, bin_error_save_dir + img_path + '.png')
        bin_pred_i = BinaryPredictionCount(5)
        bin_pred_i.append(pred, mask)
        bin_recall_str_i, bin_prec_str_i = bin_pred_i.verbose()
        print('%s. %s. %s' % (img_path, bin_recall_str_i, bin_prec_str_i))
    bin_recall_str, bin_prec_str = bin_pred.verbose()
    bin_recall_v2_str, bin_prec_v2_str = bin_pred_v2.verbose()
    print(model_path)
    print('Binary prediction:')
    print('  %s' % bin_recall_str)
    print('  %s' % bin_prec_str)
    print('Binary prediction with 1 - prob[0]:')
    print('  %s' % bin_recall_v2_str)
    print('  %s' % bin_prec_v2_str)


def grid_main():
    for model_name in [backbone]:
        # for (lr_step_values, max_epoch) in [([4, 6, 7], 8)]:    # , ([6, 9, 11], 12), ([10, 15, 18], 20)]:
        for (lr_step_values, max_epoch) in [([16, 25, 30], 32)]:
            main(model_name, lr_step_values, max_epoch)


if __name__ == '__main__':
    # grid_main()
    generate_bad_cases(backbone)

