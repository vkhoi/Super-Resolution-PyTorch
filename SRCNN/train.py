import argparse
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from math import log10
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data import get_training_set, get_val_set
from metrics import AverageMeter
from model import SRCNN
from utilities import compute_psnr


def validate(i, val_dataloader, model, criterion, val_loss_meter,
             val_psnr_meter, writer, config):
    """Assume batch size when doing validation is 1. 
    """
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(val_dataloader):
            input, target = batch[0].to(device), batch[1].to(device)

            output = model(input)
            loss = criterion(output, target)
            val_loss_meter.update(loss)

            output_image = torch.clamp(output[0], min=0, max=1)
            output_image = (output_image.cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
            target_image = (target[0].cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
            psnr = compute_psnr(output_image, target_image)
            val_psnr_meter.update(psnr)


def setup_optimizer(model, base_lr=1e-3):
    base_lr = config['training']['optimizer']['lr']
    if config['training']['optimizer']['name'] == 'sgd':
        optimizer = optim.SGD([
            { 'params': model.conv1.parameters() },
            { 'params': model.conv2.parameters() },
            { 'params': model.conv3.parameters(), 'lr': base_lr / 10 },
        ], lr=base_lr, momentum=config['training']['optimizer']['momentum'])
    else:
        optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': base_lr / 10},
            ], lr=base_lr)
    return optimizer


def setup_scheduler(optimizer, config):
    scheduler = StepLR(
        optimizer, step_size=config['training']['scheduler']['interval'],
        gamma=config['training']['scheduler']['lr_decay'])
    return scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='training config file')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    # Load config file.
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found')
    device = torch.device('cuda' if args.cuda else 'cpu')
    print('Use device:', device)

    # Use random seed.
    seed = np.random.randint(1, 10000)
    print('Random Seed: ', seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Create folder to log.
    log_dir = os.path.join(
        'runs', os.path.basename(args.config)[:-5] + '_' + str(seed))
    writer = SummaryWriter(log_dir=log_dir)

    # Create folder to store checkpoints.
    os.makedirs(
        os.path.join(config['training']['checkpoint_folder'],
                     os.path.basename(args.config)[:-5]),
        exist_ok=True)

    print('===> Loading datasets')
    sys.stdout.flush()
    train_set = get_training_set(
        img_dir=config['data']['train_root'],
        upscale_factor=config['training']['upscale_factor'],
        crop_size=config['data']['lr_crop_size'] * config['training']['upscale_factor'])
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=config['training']['batch_size'],
        shuffle=True)

    val_set = get_val_set(
        img_dir=config['data']['test_root'],
        upscale_factor=config['training']['upscale_factor'])
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=1, shuffle=False)

    print('===> Building model')
    sys.stdout.flush()
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)

    start_iter = 0
    best_val_psnr = -1

    if config['training']['resume'] != 'None':
        print('===> Reloading model')
        sys.stdout.flush()
        ckpt = torch.load(config['training']['resume'])
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_iter = ckpt['iter']
        best_val_psnr = ckpt['best_val_psnr']

    print('===> Start training')
    sys.stdout.flush()

    still_training = True
    i = start_iter
    val_loss_meter = AverageMeter()
    val_psnr_meter = AverageMeter()

    while i <= config['training']['iterations'] and still_training:
        for batch in train_dataloader:
            i += 1
            scheduler.step()
            model.train()

            input, target = batch[0].to(device), batch[1].to(device)

            output = model(input)
            optimizer.zero_grad()
            
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if i % config['training']['print_interval'] == 0:
                format_str = 'Iter [{:d}/{:d}] Loss: {:.6f}'
                print_str = format_str.format(
                    i,
                    config['training']['iterations'],
                    loss.item()
                )
                print(print_str)
                sys.stdout.flush()
                writer.add_scalar('loss/train_loss', loss.item(), i)

            if i % config['training']['val_interval'] == 0:
                validate(i, val_dataloader, model, criterion, val_loss_meter, 
                         val_psnr_meter, writer, config)

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i)
                writer.add_scalar('psnr/val_psnr', val_psnr_meter.avg, i)

                format_str = '===> Iter [{:d}/{:d}] Val_Loss: {:.6f}, Val_PSNR: {:.4f}'
                print(format_str.format(
                    i, config['training']['iterations'],
                    val_loss_meter.avg,
                    val_psnr_meter.avg
                ))
                sys.stdout.flush()

                if val_psnr_meter.avg >= best_val_psnr:
                    best_val_psnr = val_psnr_meter.avg
                    ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_psnr': best_val_psnr,
                        'iter': i
                    }
                    path = '{}/{}/{}_{}.pth'.format(
                        config['training']['checkpoint_folder'],
                        os.path.basename(args.config)[:-5],
                        os.path.basename(args.config)[:-5], i)
                    torch.save(ckpt, path)

                val_loss_meter.reset()
                val_psnr_meter.reset()

            if i >= config['training']['iterations']:
                still_training = False
                break

