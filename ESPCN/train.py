import argparse
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from math import log10
from torch.utils.data import DataLoader

from config import config
from data import get_training_set, get_val_set
from model import ESPCN


def train(epoch, train_dataloader, device, model, optimizer, critetion):
    epoch_loss = 0
    epoch_psnr = 0
    n_iterations = len(train_dataloader)

    for i, batch in enumerate(train_dataloader):
        input, target = batch[0].to(device), batch[1].to(device)

        output = model(input)
        optimizer.zero_grad()
        
        loss = criterion(output, target)
        epoch_loss += loss.item()
        
        psnr = 10 * log10(1 / loss.item())
        epoch_psnr += psnr

        loss.backward()
        optimizer.step()

        if (i + 1) % 150 == 0:
            print('Epoch[{}]({}/{}): Loss: {:.8f}, PSNR: {:.4f}'.format(
                  epoch, i, len(train_dataloader), loss.item(), psnr))
            sys.stdout.flush()

    epoch_loss /= n_iterations
    epoch_psnr /= n_iterations

    print('===> Epoch {} complete: Avg. Loss: {:.8f}, Avg. PSNR: {:.4f}'.format(
          epoch, epoch_loss, epoch_psnr))
    sys.stdout.flush()

    return epoch_loss, epoch_psnr


def test(val_dataloader, model, criterion):
    avg_psnr = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    avg_psnr /= len(val_dataloader)

    print('===> Avg. PSNR on test set: {:.4f} dB'.format(avg_psnr))
    sys.stdout.flush()

    return avg_psnr


def checkpoint(checkpoints_dir, epoch, model, optimizer, history):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
    }

    path = '{}/ckpt{}.pth'.format(checkpoints_dir, epoch)
    torch.save(ckpt, path)
    print('Checkpoint saved to {}'.format(path))
    sys.stdout.flush()


def setup_optimizer(model, base_lr=1e-3):
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': [param for param in model.conv3.parameters()], 'lr': base_lr / 10}],
        lr=base_lr)
    return optimizer


def adjust_learning_rate(optimizer, lr=None):
    if lr is None:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] /= 2
    else:
        for i, param_group in enumerate(optimizer.param_groups):
            if i < 2:
                param_group['lr'] = lr
            else:
                param_group['lr'] = lr / 10


def get_max(history):
    if len(history) == 0:
        return -1, -1
    pos = np.argmax(np.array(history))
    val = history[pos]
    return pos, val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale_factor', type=int, required=True,
                        help='super resolution upscale factor')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--use_new_lr', type=float, default=-1,
                        help='lr to use, otherwise use from config.py')
    parser.add_argument('--n_epochs', type=int, default=80,
                        help='number of epochs to train')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--reload_from', type=str, default='',
                        help='directory to checkpoint to resume training from')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='epoch number to start from')
    parser.add_argument('--seed', type=int, default=17,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found')
    device = torch.device('cuda' if args.cuda else 'cpu')
    print('Use device:', device)

    if args.use_new_lr > -1:
        base_lr = args.use_new_lr
    else:
        base_lr = config['base_lr']

    torch.manual_seed(args.seed)

    # Create folder to store checkpoints.
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print('===> Loading datasets')
    sys.stdout.flush()
    train_set = get_training_set(args.upscale_factor)
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=config['train_num_workers'])
    val_set = get_val_set(args.upscale_factor)
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=1, shuffle=False)

    print('===> Building model')
    sys.stdout.flush()
    model = ESPCN(upscale_factor=args.upscale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = setup_optimizer(model, base_lr=base_lr)
    history = {
        'train_loss': [],
        'train_psnr': [],
        'val_psnr': [],
        'last_epoch_change_lr': 0,
    }

    if len(args.reload_from) > 0:
        print('===> Reloading model from checkpoint {}'.format(
              args.reload_from))
        sys.stdout.flush()
        ckpt = torch.load(args.reload_from)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.use_new_lr > -1:
            adjust_learning_rate(optimizer, args.use_new_lr)
        history = ckpt['history']
        if 'last_epoch_change_lr' not in ckpt['history'].keys():
            history['last_epoch_change_lr'] = 0

    print('===> Start training')
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train_loss, train_psnr = train(
            epoch, train_dataloader, device, model, optimizer, criterion)
        val_psnr = test(val_dataloader, model, criterion)

        pos, max_val_psnr = get_max(history['val_psnr'])
        if val_psnr <= max_val_psnr and epoch - pos > 200:
            # If we have waited for more than 200 epochs but still got no
            # improvement, then stop.
            break

        history['train_loss'].append(train_loss)
        history['train_psnr'].append(train_psnr)
        history['val_psnr'].append(val_psnr)
        checkpoint(args.checkpoints_dir, epoch, model, optimizer, history)

        # Check to see if we should divide the lr by 2.
        if epoch >= 200 and epoch - history['last_epoch_change_lr'] >= 100:
            recent_loss = history['train_loss'][epoch-100:epoch]
            ancient_loss = history['train_loss'][:epoch-100]
            a = min(recent_loss)
            b = min(ancient_loss)
            ratio = (a - b) / b
            if ratio > 0 or abs(ratio) < 0.002:
                print('\n===> Divide lr by 2, current base lr: ', end='')
                adjust_learning_rate(optimizer)
                base_lr = optimizer.param_groups[0]['lr']
                print(base_lr)
                print('\n')

                history['last_epoch_change_lr'] = epoch

                if base_lr < 1e-5:
                    print('===> lr is less than 1e-5, exit!')
                    break
