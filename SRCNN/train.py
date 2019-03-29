import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from math import log10
from torch.utils.data import DataLoader

from data import get_training_set, get_val_set
from model import SRCNN


def train(epoch, train_dataloader, device, model, optimizer, critetion):
    epoch_loss = 0
    epoch_psnr = 0
    n_iterations = len(train_dataloader)

    for i, batch in enumerate(train_dataloader):
        input, target = batch[0].to(device), batch[1].to(device)

        output = model(input)
        optimizer.zero_grad()
        
        loss = criterion(output, target)
        epoch_loss += loss
        
        psnr = 10 * log10(1 / loss.item())
        epoch_psnr += psnr

        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print('Epoch[{}]({}/{}): Loss: {:.4f}, PSNR: {:.4f}'.format(
                  epoch, i, len(train_dataloader), loss.item(), psnr))
            sys.stdout.flush()

    print('===> Epoch {} complete: Avg. Loss: {:.4f}, Avg. PSNR: {:.4f}'.format(
          epoch, epoch_loss / n_iterations, epoch_psnr / n_iterations))
    sys.stdout.flush()


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


def checkpoint(checkpoints_dir, epoch, model, optimizer):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    path = '{}/ckpt{}.pth'.format(checkpoints_dir, epoch)
    torch.save(ckpt, path)
    print('Checkpoint saved to {}'.format(path))
    sys.stdout.flush()


def setup_optimizer(model):
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': [param for param in model.conv3.parameters()], 'lr': 1e-4}],
        lr=1e-3)
    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale_factor', type=int, required=True,
                        help='super resolution upscale factor')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=80,
                        help='number of epochs to train')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--reload_from', type=str, default='',
                        help='directory to checkpoint to resume training from')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='epoch number to start from')
    parser.add_argument('--seed', type=int, default=1742,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found')
    device = torch.device('cuda' if args.cuda else 'cpu')
    print('Use device:', device)

    torch.manual_seed(args.seed)

    # Create folder to store checkpoints.
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print('===> Loading datasets')
    sys.stdout.flush()
    train_set = get_training_set(args.upscale_factor)
    train_dataloader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_set = get_val_set(args.upscale_factor)
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=1, shuffle=False)

    print('===> Building model')
    sys.stdout.flush()
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = setup_optimizer(model)

    if len(args.reload_from) > 0:
        print('===> Reloading model from checkpoint {}'.format(
              args.reload_from))
        sys.stdout.flush()
        ckpt = torch.load(args.reload_from)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train(epoch, train_dataloader, device, model, optimizer, criterion)
        test(val_dataloader, model, criterion)
        checkpoint(args.checkpoints_dir, epoch, model, optimizer)
