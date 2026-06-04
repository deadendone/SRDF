import os
import cv2
import math



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

GPU_ID = 2                 # 指定使用哪块GPU: 0, 1, 2, 3

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

working_path = os.path.dirname(os.path.abspath(__file__))
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from utils.loss import *
from datasets import RS_ST as RS
from lib.models import SRDF as Net

NET_NAME = 'SRDF'
DATA_NAME = 'HRSCD_100e_316_3log.out'


# Training options
args = {
    'train_batch_size': 6,
    'val_batch_size': 6,
    'lr': 8e-5,
    'epochs': 100,
    'gpu': True,
    'lr_decay_power': 1.5,
    'weight_decay': 0.01,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'warmup_epochs': 3,        # 【修改】从10改为3
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])


##############################
# 【修改】带下限保护的自动权重Loss
##############################

class SafeAutomaticWeightedLoss(nn.Module):
    """
    改进的自动加权Loss，给每个loss权重设置下限
    防止某个任务的权重被压到接近0导致该任务无法学习
    """
    def __init__(self, num=5, min_weight=0.1):
        super(SafeAutomaticWeightedLoss, self).__init__()
        # 初始化为1.0而不是随机值
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.min_weight = min_weight  # 权重下限

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # 权重 = 0.5 / sigma^2，sigma^2有下限
            sigma_sq = self.params[i] ** 2
            # 确保权重不会太小: weight >= min_weight
            weight = torch.clamp(0.5 / sigma_sq, min=self.min_weight)
            reg = torch.log(1 + sigma_sq)
            loss_sum += weight * loss + reg
        return loss_sum


# 使用带保护的版本
uwl = SafeAutomaticWeightedLoss(5, min_weight=0.2)


def main():
    net = Net(64, num_classes=RS.num_classes,
              pretrained_path=r'/home/zwenbo/BGSNet-main/BGSNet-main/weights/pvt_v2_b2.pth').cuda()

    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.Data('test')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()

    # 【修改】分组学习率优化器
    optimizer = get_grouped_optimizer(net, base_lr=args['lr'])

    # AutomaticWeightedLoss 优化器
    optimizer1 = optim.AdamW(uwl.parameters(), lr=0.0001)  # 【修改】从1e-5提高到1e-4

    train(train_loader, net, criterion, optimizer, val_loader, optimizer1)
    writer.close()
    print('Training finished.')


def get_grouped_optimizer(model, base_lr=1e-4):
    """
    分组学习率:
    - backbone: 0.5x lr（预训练参数，适度微调）
    - 原有decoder: 1x lr
    - 新增模块: 2x lr（需要更快学习）
    """
    new_module_names = ['hierarchical_graph', 'diff_enhance', 'sem_refine1', 'sem_refine2']

    backbone_params = []
    original_params = []
    new_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        elif any(m in name for m in new_module_names):
            new_params.append(param)
        else:
            original_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': base_lr * 0.5, 'name': 'backbone'},       # 【修改】0.1→0.5
        {'params': original_params, 'lr': base_lr, 'name': 'original'},
        {'params': new_params, 'lr': base_lr * 2.0, 'name': 'new_modules'},
    ]

    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=args['weight_decay'])

    print(f"=== Optimizer Parameter Groups ===")
    print(f"  Backbone:  {len(backbone_params):4d} params, lr={base_lr * 0.5:.6f}")
    print(f"  Original:  {len(original_params):4d} params, lr={base_lr:.6f}")
    print(f"  New mods:  {len(new_params):4d} params, lr={base_lr * 2.0:.6f}")

    return optimizer


def train(train_loader, net, criterion, optimizer, val_loader, optimizer1):
    bestaccT = 0
    bestFscdV = 0.0
    bestloss = 1.0
    bestaccV = 0.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sc = ChangeSimilarity().cuda()
    curr_epoch = 0

    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1

            # 【修改】使用修正后的warmup + cosine学习率调度
            adjust_lr_warmup_cosine(optimizer, curr_epoch, i, len(train_loader))

            imgs_A, imgs_B, labels_A, labels_B, labels_C = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
                semantic_bd = (labels_C / 255).cuda().long()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()
            optimizer1.zero_grad()
            out_change, outputs_A, outputs_B, out_bd = net(imgs_A, imgs_B)

            assert outputs_A.size()[1] == RS.num_classes

            loss_seg1 = criterion(outputs_A, labels_A) * 0.5
            loss_seg2 = criterion(outputs_B, labels_B) * 0.5
            cretion1 = weighted_bce()
            loss_bn = cretion1(out_change, labels_bn)
            loss_edge = cretion1(out_bd, semantic_bd.float())
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)

            loss = uwl(loss_seg1, loss_seg2, loss_bn, loss_edge, loss_sc)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer1.step()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()

            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg1.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                scale_info = get_scale_info(net)
                # 【新增】打印AWL权重监控
                awl_info = get_awl_weights(uwl)
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val * 100))
                print(f'  {scale_info}')
                print(f'  {awl_info}')

                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)

        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if Fscd_v > bestFscdV:
            bestFscdV = Fscd_v
            bestaccV = acc_v
            bestloss = loss_v
            torch.save(net.state_dict(),
                       os.path.join(args['chkpt_dir'], NET_NAME + '_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth' \
                                    % (curr_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100)))
        print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f' % (
            time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B, labels_C = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()
            semantic_bd = (labels_C / 255).cuda().long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B, out_bd = net(imgs_A, imgs_B)
            cretion1 = weighted_bce()
            loss = cretion1(out_change, labels_bn)

        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def get_scale_info(net):
    """获取新增模块的scale参数"""
    info = ""
    if hasattr(net, 'hierarchical_graph'):
        info += f"[Scales] graph_low={net.hierarchical_graph.low_scale.item():.4f} "
        info += f"graph_high={net.hierarchical_graph.high_scale.item():.4f} "
    if hasattr(net, 'diff_enhance'):
        info += f"diff={net.diff_enhance.enhance_scale.item():.4f} "
    if hasattr(net, 'sem_refine1'):
        info += f"ref1={net.sem_refine1.refine_scale.item():.4f} "
        info += f"ref2={net.sem_refine2.refine_scale.item():.4f}"
    return info


def get_awl_weights(awl):
    """
    【新增】打印AutomaticWeightedLoss各任务的实际权重
    用于监控是否有某个任务权重被压到0
    """
    weights = []
    for i in range(len(awl.params)):
        sigma_sq = awl.params[i].item() ** 2
        w = max(0.5 / sigma_sq, awl.min_weight)
        weights.append(w)
    names = ['seg1', 'seg2', 'bn', 'edge', 'sc']
    info = "[AWL weights] " + " ".join([f"{n}={w:.3f}" for n, w in zip(names, weights)])
    return info


def adjust_lr_warmup_cosine(optimizer, epoch, batch_idx, batches_per_epoch):
    """
    【修正】Warmup + Cosine 学习率调度
    - warmup: 3个epoch，从 0.1x 线性增长到 1x（不再从0.01x开始）
    - cosine: 余弦退火到 0.01x
    """
    warmup_epochs = args['warmup_epochs']
    total_epochs = args['epochs']

    current_progress = epoch + batch_idx / batches_per_epoch

    if current_progress < warmup_epochs:
        # warmup: 从0.1倍增长到1倍（而不是从0.01倍）
        scale = 0.1 + 0.9 * (current_progress / warmup_epochs)
    else:
        # cosine退火: 从1倍降到0.01倍
        progress = (current_progress - warmup_epochs) / (total_epochs - warmup_epochs)
        scale = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 各组的base_lr
    base_lrs = {
        'backbone': args['lr'] * 0.5,       # 【修改】0.1→0.5
        'original': args['lr'],
        'new_modules': args['lr'] * 2.0
    }

    for param_group in optimizer.param_groups:
        group_name = param_group.get('name', 'original')
        base_lr = base_lrs.get(group_name, args['lr'])
        param_group['lr'] = base_lr * scale


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


if __name__ == '__main__':
    main()