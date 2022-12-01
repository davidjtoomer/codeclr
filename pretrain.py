import argparse
import logging
import os

import numpy as np
import torch
from torch.utils import tensorboard

from graph_code_embedding.cass import CassConfig
from graph_code_embedding.data import train_val_test_split
from graph_code_embedding.model import ContrastiveLearner


logging.basicConfig(
    format='[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                    help='The top-level preprocessed data directory.')
parser.add_argument('--benchmark', type=int, default=1000,
                    choices=[1000, 1400], help='The benchmark number.')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='The directory in which to store the logs.')
parser.add_argument('--save_interval', type=int, default=1,
                    help='The number of epochs between saving the model.')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='The number of epochs for which to train.')
parser.add_argument('--batch_size', type=int,
                    default=128, help='The batch size.')
parser.add_argument('--train_frac', type=float, default=0.6,
                    help='The fraction of the data to use for training.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='The learning rate.')
parser.add_argument(
    '--mask_frac',
    type=float,
    default=0.25,
    help='The fraction of nodes to mask for data augmentation.')
# CASS configuration
parser.add_argument(
    '--annot_mode',
    type=int,
    default=2,
    choices=[0, 1, 2],
    help='CASS configuration: node prefix label. 0: No change. 1: Add a prefix to each internal nodel label. 2: Add a prefix to parenthesis node label.')
parser.add_argument(
    '--compound_mode',
    type=int,
    default=1,
    choices=[0, 1, 2],
    help='CASS configuration: compound statements. 0: No change. 1: Drop all features relevant to compound statements. 2: Replace with "{#}".')
parser.add_argument(
    '--gfun_mode',
    type=int,
    default=1,
    choices=[0, 1, 2],
    help='CASS configuration: global functions. 0: No change. 1: Drop all features relevant to global functions. 2: Drop function identifier and replace with "#EXFUNC".')
parser.add_argument(
    '--gvar_mode',
    type=int,
    default=3,
    choices=[0, 1, 2, 3],
    help='CASS configuration: global variables. 0: No change. 1: Drop all features relevant to global variables. 2: Replace with "$GVAR". 3: Replace with "$VAR".')
parser.add_argument(
    '--fsig_mode',
    type=int,
    default=1,
    choices=[0, 1],
    help='CASS configuration: function I/O cardinality. 0: No change. 1: Include the input and output cardinality per function in GAT.')
args = parser.parse_args()

config = CassConfig(
    annot_mode=args.annot_mode,
    compound_mode=args.compound_mode,
    gfun_mode=args.gfun_mode,
    gvar_mode=args.gvar_mode,
    fsig_mode=args.fsig_mode)

parameter_tag = f'{config.tag}_mask_frac={args.mask_frac}_batch_size={args.batch_size}_lr={args.lr}'
LOG_DIR = os.path.join(args.log_dir, parameter_tag)
os.makedirs(LOG_DIR, exist_ok=True)
writer = tensorboard.SummaryWriter(LOG_DIR)

logger.info('Loading data...')
ALL_DATA_DIR = os.path.join(
    args.data_dir,
    f'Project_CodeNet_C++{args.benchmark}',
    config.tag,
    'all')

if not os.path.exists(ALL_DATA_DIR):
    logger.error(f'Data directory does not exist: {ALL_DATA_DIR}')
    exit(1)

train_dataloader, val_dataloader, test_dataloader = train_val_test_split(
    ALL_DATA_DIR,
    train_frac=args.train_frac,
    batch_size=args.batch_size)
logger.info('Successfully loaded data.')

VOCAB_FILE = os.path.join(
    args.data_dir,
    f'Project_CodeNet_C++{args.benchmark}',
    config.tag,
    'vocab.pt')

if os.path.exists(VOCAB_FILE):
    logger.info(f'Loading vocabulary from {VOCAB_FILE}')
    vocab = torch.load(VOCAB_FILE)
    vocab_size = len(vocab)
else:
    logger.info('No vocabulary found. Using default embedding size of 1000.')
    vocab_size = 1000

gcn_layers = [128, 128, 64, 32]
model = ContrastiveLearner(
    gcn_layers, vocab_size=vocab_size, mask_frac=args.mask_frac)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

logger.info('Training...')
for epoch in range(args.num_epochs):
    epoch_train_losses = []
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())
        batch_iteration = epoch * len(train_dataloader) + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), batch_iteration)

    epoch_train_loss = np.mean(epoch_train_losses)
    writer.add_scalar('train/loss', loss.item(), epoch)

    with torch.no_grad():
        epoch_val_losses = []
        for batch in val_dataloader:
            loss = model(batch)
            epoch_val_losses.append(loss.item())

        epoch_val_loss = np.mean(epoch_val_losses)
        writer.add_scalar('val/loss', loss.item(), epoch)

    if epoch % args.save_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(LOG_DIR, f'checkpoint_{epoch}.pt'))
