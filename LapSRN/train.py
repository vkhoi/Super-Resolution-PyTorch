import argparse
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
import yaml

from math import log10
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data import get_training_set, get_val_set
from loss import L1_Charbonnier_loss
from metrics import AverageMeter
from model import LapSRN
from utilities import compute_psnr


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def validate(i, val_dataloader, model, criterion, val_loss_meters,
             val_psnr_meters, writer, config):
    """Assume batch size when doing validation is 1. 
    """
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(val_dataloader):
            input, targets = batch[0].to(device), batch[1]
            for _ in range(len(targets)):
                targets[_] = targets[_].to(device)

            outputs = model(input)

            # Record loss at each level.
            for _ in range(len(targets)):
                loss = criterion(outputs[_], targets[_])
                val_loss_meters[_].update(loss)

                output_image = torch.clamp(outputs[_][0], min=0, max=1)
                output_image = (output_image.cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                target_image = (targets[_][0].cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
                psnr = compute_psnr(output_image, target_image)
                val_psnr_meters[_].update(psnr)


def setup_optimizer(model, config):
    lr = config['training']['optimizer']['lr']
    weight_decay = config['training']['optimizer']['weight_decay']
    if config['training']['optimizer']['name'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=lr,
            momentum=config['training']['optimizer']['momentum'],
            weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
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
                     os.path.basename(args.config)[:-5] + '_' + str(seed)),
        exist_ok=True)

    print('===> Loading datasets')
    sys.stdout.flush()
    train_set = get_training_set(
        img_dir=config['data']['train_root'],
        upscale_factor=config['model']['upscale_factor'],
        img_channels=config['model']['img_channels'],
        crop_size=config['data']['hr_crop_size'])
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=config['training']['batch_size'],
        shuffle=True)

    val_set = get_val_set(
        img_dir=config['data']['test_root'],
        upscale_factor=config['model']['upscale_factor'],
        img_channels=config['model']['img_channels'])
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=1, shuffle=False)

    print('===> Building model')
    sys.stdout.flush()
    model = LapSRN(img_channels=config['model']['img_channels'],
                   upscale_factor=config['model']['upscale_factor'],
                   n_feat=config['model']['n_feat'],
                   n_recursive=config['model']['n_recursive'],
                   local_residual=config['model']['local_residual']).to(device)
    # criterion = nn.MSELoss()
    criterion = L1_Charbonnier_loss()
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
    n_levels = int(np.log2(config['model']['upscale_factor']))
    val_loss_meters = []
    val_psnr_meters = []
    for _ in range(n_levels):
        val_loss_meters.append(AverageMeter())
        val_psnr_meters.append(AverageMeter())

    while i <= config['training']['iterations'] and still_training:
        for batch in train_dataloader:
            i += 1
            scheduler.step()
            model.train()

            input, targets = batch[0].to(device), batch[1]
            for _ in range(n_levels):
                targets[_] = targets[_].to(device)

            outputs = model(input)
            optimizer.zero_grad()

            # Record loss at each level for plotting to TensorBoard.
            losses = [criterion(outputs[0], targets[0])]
            for _ in range(1, len(targets)):
                losses.append(criterion(outputs[_], targets[_]))

            # Accumulate.
            loss = losses[0]
            for _ in range(1, n_levels):
                loss = loss + losses[_]

            loss.backward()
            optimizer.step()

            if i % config['training']['print_interval'] == 0:
                print('Iter [{:d}/{:d}]'.format(i, config['training']['iterations']), end='')
                for _ in range(n_levels):
                    level = 2**(_ + 1)
                    print(' Loss_{:d}x: {:.6f}'.format(level, losses[_].item()), end='')
                    writer.add_scalar('loss/train_loss_{:d}x'.format(level), losses[_].item(), i)
                print('')
                sys.stdout.flush()

            if i % config['training']['val_interval'] == 0:
                validate(i, val_dataloader, model, criterion, val_loss_meters, 
                         val_psnr_meters, writer, config)

                print('Iter [{:d}/{:d}]'.format(i, config['training']['iterations']), end='')
                for _ in range(n_levels):
                    level = 2**(_ + 1)
                    print(' Loss_{:d}x: {:.6f}'.format(level, val_loss_meters[_].avg), end='')
                    print(' PSNR_{:d}x: {:.6f}'.format(level, val_psnr_meters[_].avg), end='')
                    writer.add_scalar('loss/val_loss_{:d}x'.format(level), val_loss_meters[_].avg, i)
                    writer.add_scalar('psnr/val_psnr_{:d}x'.format(level), val_psnr_meters[_].avg, i)
                print('')
                sys.stdout.flush()

                if val_psnr_meters[-1].avg >= best_val_psnr:
                    best_val_psnr = val_psnr_meters[-1].avg
                    ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_psnr': best_val_psnr,
                        'iter': i
                    }
                    path = '{}/{}/{}_{}.pth'.format(
                        config['training']['checkpoint_folder'],
                        os.path.basename(args.config)[:-5] + '_' + str(seed),
                        os.path.basename(args.config)[:-5], i)
                    torch.save(ckpt, path)

                for _ in range(n_levels):
                    val_loss_meters[_].reset()
                    val_psnr_meters[_].reset()

            if i >= config['training']['iterations']:
                still_training = False
                break
