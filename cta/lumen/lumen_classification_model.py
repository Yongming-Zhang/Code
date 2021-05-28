from lumen_util import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import shutil
import numpy
import logging
import logging.config
from collections import defaultdict
from lumen_dataset import LumenListDataset
from lib.models.util import adjust_learning_rate
from lib.utils.mio import create_log_config, mkdir_safe, load_string_list, save_string_list
from lib.utils.meter import AverageMeter, AUCMeter, recall_precision, accuracy, confusion_matrix
import time


feat_dim = 256
backbone = 'resnet18_drop_l4c%d' % feat_dim
list_dir = '/breast_data/cta/new_data/b2_sg_407/model_data/lumen/lists/new_p32x32x32_w300-700_-100-700_-500-1000/'
train_list_path = list_dir + 'train_balance_label11111.txt'
valid_list_path = list_dir + 'valid_img_list.txt'
train_batch_size = 32
base_lr, snapshot = 0.01, 1
gpu_id = int(sys.argv[1])
num_classes = 5
is_pretrained = True
center_crop_z, random_crop_z, center_crop_xy, random_crop_xy, resize_xy = 32, 28, 23, 20, 40
torch.cuda.manual_seed(1)
torch.cuda.set_device(gpu_id)
label_dict = ['fp', 'low', 'cal', 'mix', 'ste']


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
        self.backbone = models.resnet18(pretrained=True)
        self.conv3d_1 = conv3d(256 if 112 >= resize_xy > 56 else 128, feat_dim, (7, 1, 1), (2, 1, 1), (3, 0, 0))
        self.conv3d_2 = conv3d(feat_dim, feat_dim, (3, 1, 1), (2, 1, 1), (1, 0, 0))
        if random_crop_z > 28:
            z_ds = 8
            self.conv3d_2_2 = conv3d(feat_dim, feat_dim, (3, 3, 3), (2, 1, 1), (1, 1, 1))
        else:
            z_ds = 4
        self.conv3d_3 = conv3d(feat_dim, feat_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv3d_4 = conv3d(feat_dim, feat_dim, (random_crop_z // z_ds, 1, 1), (1, 1, 1), (0, 0, 0))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(feat_dim, num_classes)
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
        if 112 >= resize_xy > 56:
            x = self.backbone.layer3(x)
        # print('layer2:', x.size())
        bd, c, h, w = x.size()
        x = x.view(b, d, c, h, w).transpose(1, 2).contiguous()
        return x

    def forward(self, x):
        x = self._forward_backbone(x)
        # print('backbone:', x.size())
        x = self.conv3d_1(x)
        # print('conv3d_1:', x.size())
        x = self.conv3d_2(x)
        if random_crop_z > 28:
            x = self.conv3d_2_2(x)
        # print('conv3d_2:', x.size())
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        # print('conv3d_3:', x.size())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def main(model_name, lr_step_values, max_epoch):
    model_save_dir = train_list_path[:-4] + '/models/%s/' % model_name
    size_str = 'crop%dx%dx%d_resize%d' % (random_crop_z, random_crop_xy, random_crop_xy, resize_xy)
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
    random_crop_train = (random_crop_z, random_crop_xy, random_crop_xy)
    resize2d = (resize_xy, resize_xy)
    train_dataset = LumenListDataset(train_list_path, (0, 360), center_crop_train, random_crop_train, True, resize2d)
    logger.info('Train_list: ' + train_list_path)

    center_crop_valid = (random_crop_z, random_crop_xy, random_crop_xy)
    test_dataset = LumenListDataset(valid_list_path, None, center_crop_valid, None, None, resize2d)
    logger.info('Test_list: ' + valid_list_path)
    image, label = test_dataset[0]
    print(image.size(), label)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = LumenNet()
    model = model.cuda()
    params = model.parameters()
    logger.info('Parameters: {}'.format(sum([p.data.nelement() for p in params])))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), base_lr, momentum=0.9, nesterov=True, weight_decay=0.0001)

    test(test_loader, model, criterion, logger)
    for epoch in range(max_epoch):
        cur_lr = adjust_learning_rate(base_lr, lr_step_values, optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, cur_lr, logger)
        test(test_loader, model, criterion, logger)
        if (epoch + 1) % snapshot == 0 or epoch == (max_epoch - 1):
            state = model.state_dict()
            save_path = model_save_prefix + '_epoch%d.pkl' % (epoch + 1)
            torch.save(state, save_path)
            logger.info('Snapshot: ' + save_path)


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
    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        images = images.cuda()
        batch_size = images.size(0)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        probs = F.softmax(output, dim=1).data
        all_probs.append(probs)
        all_labels.append(target.view(batch_size, 1))
        # measure accuracy and record loss
        acc1 = accuracy(output.data, target, topk=(1,))[0]
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


def get_instance_probs(probs, path_size_labels):
    instance_probs = defaultdict(list)
    for i, (path, size, label) in enumerate(path_size_labels):
        psid, label_name, img_name = path.split('/')[-3:]
        instance_name = psid + '/' + label_name + '/' + img_name.split('_')[0]
        instance_probs[instance_name].append(probs[i])
    return instance_probs


def cmp_instance_level_eval(probs, path_size_labels):
    instance_probs = get_instance_probs(probs, path_size_labels)


def test(loader, model, criterion, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
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
    for i, (images, target) in enumerate(loader):
        target = target.cuda(async=True)
        images = images.cuda(async=True)
        if torch.__version__[:3] <= '0.3':
            images = torch.autograd.Variable(images, volatile=True)
            target = torch.autograd.Variable(target, volatile=True)
        batch_size = images.size(0)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        probs = F.softmax(output, dim=1).data
        all_probs.append(probs)
        all_labels.append(target.view(batch_size, 1))
        # measure accuracy and record loss
        acc1 = accuracy(output.data, target, topk=(1,))[0]
        accuracies.update(acc1 / 100., batch_size)
        losses.update(loss.item() if torch.__version__[:3] >= '0.4' else loss.data[0], batch_size)
        probs_numpy = probs.cpu().numpy()
        pos_scores = 1 - probs_numpy[:, 0]
        labels = target.cpu().numpy()
        bin_labels = numpy.int32(labels > 0)
        bin_preds = numpy.int32(probs_numpy.argmax(axis=1) > 0)
        for cid in range(num_classes):
            correct_count, gt_count, pred_count = recall_precision(probs, target, cid)
            correct_count_bin = numpy.sum((labels == cid) * (bin_preds == (cid > 0)))
            recalls[cid].update(correct_count / (gt_count + 1e-5), gt_count)
            bin_recalls[cid].update(correct_count_bin / (gt_count + 1e-5), gt_count)
            precisions[cid].update(correct_count / (pred_count + 1e-5), pred_count)
        bin_correct_num += numpy.sum(bin_labels == bin_preds)
        bin_total_num += batch_size
        bin_correct_pos_num += numpy.sum(bin_labels * bin_preds)
        bin_total_pos_num += numpy.sum(bin_labels)
        bin_pred_pos_num += numpy.sum(bin_preds)
        bin_auc_meter.update(pos_scores, target > 0)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    line = 'Test: Time=%.3fs/batch, Loss=%.6f, Accuracy=%.2f%%, Details:' % (batch_time.avg, losses.avg, accuracies.avg * 100.)
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
        msg = '    Class_%d: recall=%.2f%%, bin_recall=%.2f%%, precision=%.2f%%, pos_count=%d, ' % (
            cid, recall, recall_bin, prec, pos_count)
        msg += ', '.join(['pred_%d=%04d' % (cj, cm[cid, cj]) for cj in range(num_classes)])
        logger.info(msg)


def get_rate_str(find_num, total_num):
    rate = find_num * 100. / (total_num + 1e-5)
    return '%.2f%%(%d/%d)' % (rate, find_num, total_num)


def generate_bad_cases(model_name, model_path=None, save_bad=True):
    random_crop_z_, center_crop_xy_, random_crop_xy_, resize_xy_ = 28, 23, 20, 40
    max_epoch, model_epoch = 12, 4
    set_name = 'valid_new'
    model_save_dir = train_list_path[:-4] + '/models/%s/' % model_name
    size_str = 'crop%dx%dx%d_resize%d' % (random_crop_z_, random_crop_xy_, random_crop_xy_, resize_xy_)
    model_save_prefix = model_save_dir + 'ME%d_B%d%s_rr_flip_%s_lr%s' % (
        max_epoch, train_batch_size, '_ft' if is_pretrained else '', size_str, str(base_lr))
    model_path = model_save_prefix + '_epoch%d.pkl' % model_epoch
    model_path = model_path.replace('new_p', 'p')
    base_save_dir = model_path[:-4] + '/'
    save_dir = base_save_dir + '%s_bad/' % set_name
    bin_error_save_dir, type_error_save_dir = save_dir + 'bin_error/', save_dir + 'type_error/'
    cm_save_dir = save_dir + 'cm/'
    mkdir_safe(bin_error_save_dir)
    mkdir_safe(type_error_save_dir)
    mkdir_safe(cm_save_dir)
    image_list_path = valid_list_path if set_name == 'valid_new' else train_list_path
    center_crop_valid = (random_crop_z_, random_crop_xy_, random_crop_xy_)
    test_dataset = LumenListDataset(image_list_path, None, center_crop_valid, None, None, (resize_xy_, resize_xy_))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = LumenNet()
    model = model.cuda()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    all_probs, all_labels = [], []
    for i, (images, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        images = images.cuda(async=True)
        if torch.__version__[:3] <= '0.3':
            images = torch.autograd.Variable(images, volatile=True)
            target = torch.autograd.Variable(target, volatile=True)
        batch_size = images.size(0)
        # compute output
        output = model(images)
        probs = F.softmax(output, dim=1).data
        all_probs.append(probs)
        all_labels.append(target.view(batch_size, 1))
    all_probs = torch.cat(all_probs).cpu().numpy()
    # instance_probs = get_instance_probs(all_probs, test_loader.dataset.path_size_labels)
    lines = []
    bin_error_num, all_error_num = 0, 0
    p_find_nums, p_bin_find_nums, p_total_nums = [0] * 5, [0] * 5, [0] * 5
    g_find_nums, g_bin_find_nums, g_total_nums = [0] * 5, [0] * 5, [0] * 5
    p_instance_find_flags = [{} for i in range(5)]
    for i, (path, size, label) in enumerate(test_loader.dataset.path_size_labels):
        pred = all_probs[i].argmax()
        pred_bin, label_bin = int(pred > 0), int(label > 0)
        psid, label_name, img_name = path.split('/')[-3:]
        line = psid + ' ' + img_name + ' gt=' + label_name + ' pred=%s ' % label_dict[pred]
        line += '/'.join(['%s=%.4f' % (label_dict[j], all_probs[i, j]) for j in range(5)])
        lines.append(line)
        instance_name = psid + '/' + label_name + '/' + img_name.split('_')[0]
        if img_name[0] == 'P':
            if instance_name not in p_instance_find_flags[label]:
                p_instance_find_flags[label][instance_name] = False if label > 0 else True
            if pred_bin == label_bin:
                p_bin_find_nums[label] += 1
                if label > 0:
                    p_instance_find_flags[label][instance_name] = True
            if label == 0 and pred_bin != label_bin:
                p_instance_find_flags[label][instance_name] = False
            if pred == label:
                p_find_nums[label] += 1
            p_total_nums[label] += 1
        else:
            if pred_bin == label_bin:
                g_bin_find_nums[label] += 1
            if pred == label:
                g_find_nums[label] += 1
            g_total_nums[label] += 1
        if pred != label:
            save_name = psid + '_' + img_name[:-4] + '_gt=%s_%.4f_pred=%s_%.4f.png' % (
                label_name, all_probs[i, label], label_dict[pred], all_probs[i, pred])
            cur_save_dir = type_error_save_dir if pred > 0 and label > 0 else bin_error_save_dir
            if pred == 0 or label == 0:
                bin_error_num += 1
            all_error_num += 1
            shutil.copy(path, cur_save_dir + save_name)
            cur_save_dir = cm_save_dir + 'gt_%s_pred_%s/' % (label_name, label_dict[pred])
            mkdir_safe(cur_save_dir)
            shutil.copy(path, cur_save_dir + save_name)
    lines.sort()
    for label in range(1, 5):
        for instance_name in p_instance_find_flags[label].keys():
            if instance_name in p_instance_find_flags[0]:
                flag = p_instance_find_flags[0].pop(instance_name)
            if not p_instance_find_flags[label][instance_name]:
                print('%s is missed, label=%s' % (instance_name, label_dict[label]))
    print(model_path)
    print('bin_error_num=%d, all_error_num=%d' % (bin_error_num, all_error_num))
    print('P class5: %s' % (', '.join(['%s=%s' % (label_dict[i], get_rate_str(p_find_nums[i], p_total_nums[i])) for i in range(5)])))
    print('P binary: %s' % (', '.join(['%s=%s' % (label_dict[i], get_rate_str(p_bin_find_nums[i], p_total_nums[i])) for i in range(5)])))
    print('G class5: %s' % (', '.join(['%s=%s' % (label_dict[i], get_rate_str(g_find_nums[i], g_total_nums[i])) for i in range(5)])))
    print('G binary: %s' % (', '.join(['%s=%s' % (label_dict[i], get_rate_str(g_bin_find_nums[i], g_total_nums[i])) for i in range(5)])))
    print('P instance binary: %s' % (', '.join(['%s=%s' % (
        label_dict[i], get_rate_str(sum(p_instance_find_flags[i].values()), len(p_instance_find_flags[i]))) for i in range(5)])))
    save_string_list(save_dir + 'instance_probs.txt', lines)


def grid_main():
    for model_name in [backbone]:
        # for (lr_step_values, max_epoch) in [([4, 6, 7], 8)]:    # , ([6, 9, 11], 12), ([10, 15, 18], 20)]:
        for (lr_step_values, max_epoch) in [([6, 9, 11], 12)]:
            main(model_name, lr_step_values, max_epoch)


if __name__ == '__main__':
    # grid_main()
    generate_bad_cases(backbone)

