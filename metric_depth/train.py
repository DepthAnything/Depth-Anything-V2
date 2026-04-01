import argparse
import logging
import os
import pprint
import random
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

def parse_arguments():
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti'])
    parser.add_argument('--img-size', default=518, type=int)
    parser.add_argument('--min-depth', default=0.001, type=float)
    parser.add_argument('--max-depth', default=20, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--bs', default=2, type=int)
    parser.add_argument('--lr', default=0.000005, type=float)
    parser.add_argument('--pretrained-from', type=str)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    return parser.parse_args()

def setup_logging(args):
    warnings.simplefilter('ignore', np.RankWarning)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    return logger

def initialize_training(args, logger):
    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    else:
        writer = None
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    return rank, writer

def get_datasets_and_loaders(args, size):
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    else:
        raise NotImplementedError

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    return trainloader, valloader

def initialize_model(args, local_rank):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True)
    
    return model

def setup_optimizer(args, model):
    return AdamW([
        {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
        {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

def train_and_evaluate(args, logger, writer, rank, model, criterion, optimizer, trainloader, valloader, total_iters):
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info(f'===========> Epoch: {epoch}/{args.epochs}, d1: {previous_best["d1"]:.3f}, d2: {previous_best["d2"]:.3f}, d3: {previous_best["d3"]:.3f}')
            logger.info(f'===========> Epoch: {epoch}/{args.epochs}, abs_rel: {previous_best["abs_rel"]:.3f}, sq_rel: {previous_best["sq_rel"]:.3f}, rmse: {previous_best["rmse"]:.3f}, rmse_log: {previous_best["rmse_log"]:.3f}, log10: {previous_best["log10"]:.3f}, silog: {previous_best["silog"]:.3f}')
        
        trainloader.sampler.set_epoch(epoch + 1)
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            
            if random.random() < 0.5:
                img, depth, valid_mask = img.flip(-1), depth.flip(-1), valid_mask.flip(-1)
            
            pred = model(img)
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            iters = epoch * len(trainloader) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info(f'Iter: {i}/{len(trainloader)}, LR: {optimizer.param_groups[0]["lr"]:.7f}, Loss: {loss.item():.3f}')
        
        evaluate_model(args, logger, writer, rank, model, criterion, valloader, epoch, previous_best)

def evaluate_model(args, logger, writer, rank, model, criterion, valloader, epoch, previous_best):
    model.eval()
    results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()
    
    for i, sample in enumerate(valloader):
        img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
        
        with torch.no_grad():
            pred = model(img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        
        if valid_mask.sum() < 10:
            continue
        
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
        
        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1
    
    torch.distributed.barrier()
    
    for k in results.keys():
        dist.reduce(results[k], dst=0)
    dist.reduce(nsamples, dst=0)
    
    if rank == 0:
        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
        logger.info('==========================================================================================')
        
        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
    
    for k in results.keys():
        if k in ['d1', 'd2', 'd3']:
            previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        else:
            previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
    
    if rank == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

def main():
    args = parse_arguments()
    logger = setup_logging(args)
    rank, writer = initialize_training(args, logger)
    local_rank = int(os.environ["LOCAL_RANK"])
    
    trainloader, valloader = get_datasets_and_loaders(args, size=(args.img_size, args.img_size))
    model = initialize_model(args, local_rank)
    criterion = SiLogLoss().cuda(local_rank)
    optimizer = setup_optimizer(args, model)
    
    total_iters = args.epochs * len(trainloader)
    
    train_and_evaluate(args, logger, writer, rank, model, criterion, optimizer, trainloader, valloader, total_iters)

if __name__ == '__main__':
    main()
