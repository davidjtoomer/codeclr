import argparse
import logging
import os

import torch
from torchtext.vocab import build_vocab_from_iterator
import tqdm

from graph_code_embedding.cass import CassConfig, cass_tree_to_graph, load_file

logging.basicConfig(
    format='[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--benchmark',
    type=int,
    default=[1000],
    nargs='+',
    choices=[
        1000,
        1400],
    help='The benchmark number.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='data',
    help='The name of the directory that stores the data from download_data.py.')
parser.add_argument(
    '--output_dir',
    type=str,
    default='data/preprocessed',
    help='The name of the directory in which to store the preprocessed data.')
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

if not os.path.exists(args.data_dir):
    logger.error(f'Data directory {args.data_dir} does not exist.')
    exit(1)


def yield_tokens(file_path: str, config: CassConfig = None):
    for directory in tqdm.tqdm(os.listdir(file_path), leave=False):
        if os.path.isdir(os.path.join(file_path, directory)):
            for file in os.listdir(os.path.join(file_path, directory)):
                cass_trees = load_file(
                    os.path.join(file_path, directory, file), config=config)
                nodes = []
                [nodes.extend(cass_tree.nodes) for cass_tree in cass_trees]
                for node in nodes:
                    yield node.n


for benchmark in args.benchmark:
    DIRECTORY_NAME = f'Project_CodeNet_C++{benchmark}'
    DATA_DIR = os.path.join(args.data_dir, DIRECTORY_NAME, 'cass')
    if not os.path.exists(DATA_DIR):
        logger.error(
            f'Data directory {DATA_DIR} does not exist. Could not preprocess data for {benchmark}.')
        continue

    config = CassConfig(
        annot_mode=args.annot_mode,
        compound_mode=args.compound_mode,
        gfun_mode=args.gfun_mode,
        gvar_mode=args.gargs.var_mode,
        fsig_mode=args.fsig_mode)
    logger.info(
        f'Preprocessing {benchmark} with {config.tag}...')
    PREPROCESSED_DIR = os.path.join(
        args.output_dir, DIRECTORY_NAME, config.tag)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    logger.info(f'Generating vocabulary...')
    vocab = build_vocab_from_iterator(
        yield_tokens(DATA_DIR, config), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    logger.info(f'Vocabulary size: {len(vocab)}')
    torch.save(vocab, os.path.join(
        PREPROCESSED_DIR, 'vocab.pt'))

    for directory in tqdm.tqdm(
            os.listdir(DATA_DIR), leave=False):
        if os.path.isdir(
            os.path.join(
                DATA_DIR,
                directory)):
            OUTPUT_DIR_ALL = os.path.join(
                PREPROCESSED_DIR, 'all')
            OUTPUT_DIR_SORTED = os.path.join(
                PREPROCESSED_DIR, directory)
            os.makedirs(OUTPUT_DIR_ALL, exist_ok=True)
            os.makedirs(OUTPUT_DIR_SORTED, exist_ok=True)

            for filename in os.listdir(
                    os.path.join(DATA_DIR, directory)):
                if filename.endswith('.cas'):
                    cass_trees = load_file(os.path.join(
                        DATA_DIR, directory, filename), config)
                    dense_graph = cass_tree_to_graph(
                        cass_trees, vocabulary=vocab)
                    dense_graph.save(
                        os.path.join(
                            OUTPUT_DIR_SORTED, filename.replace(
                                '.cas', '.pt')))
                    dense_graph.save(
                        os.path.join(
                            OUTPUT_DIR_ALL,
                            f'{directory}_{filename.replace(".cas", ".pt")}'))
