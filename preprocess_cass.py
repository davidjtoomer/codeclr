import argparse
import logging
import os

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
    default=[0],
    nargs='+',
    choices=[
        0,
        1,
        2],
    help='CASS configuration: node prefix label. 0: No change. 1: Add a prefix to each internal nodel label. 2: Add a prefix to parenthesis node label.')
parser.add_argument(
    '--compound_mode',
    type=int,
    default=[0],
    nargs='+',
    choices=[
        0,
        1,
        2],
    help='CASS configuration: compound statements. 0: No change. 1: Drop all features relevant to compound statements. 2: Replace with "{#}".')
parser.add_argument(
    '--gfun_mode',
    type=int,
    default=[0],
    nargs='+',
    choices=[
        0,
        1,
        2],
    help='CASS configuration: global functions. 0: No change. 1: Drop all features relevant to global functions. 2: Drop function identifier and replace with "#EXFUNC".')
parser.add_argument(
    '--gvar_mode',
    type=int,
    default=[0],
    nargs='+',
    choices=[
        0,
        1,
        2,
        3],
    help='CASS configuration: global variables. 0: No change. 1: Drop all features relevant to global variables. 2: Replace with "$GVAR". 3: Replace with "$VAR".')
parser.add_argument(
    '--fsig_mode',
    type=int,
    default=[0],
    nargs='+',
    choices=[
        0,
        1],
    help='CASS configuration: function I/O cardinality. 0: No change. 1: Include the input and output cardinality per function in GAT.')
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    logger.error(f'Data directory {args.data_dir} does not exist.')
    exit(1)

for benchmark in args.benchmark:
    DIRECTORY_NAME = f'Project_CodeNet_C++{benchmark}'
    DATA_DIR = os.path.join(args.data_dir, DIRECTORY_NAME, 'cass')
    if not os.path.exists(DATA_DIR):
        logger.error(
            f'Data directory {DATA_DIR} does not exist. Could not preprocess data for {benchmark}.')
        continue

    for annot_mode in args.annot_mode:
        for compound_mode in args.compound_mode:
            for gfun_mode in args.gfun_mode:
                for gvar_mode in args.gvar_mode:
                    for fsig_mode in args.fsig_mode:
                        config = CassConfig(
                            annot_mode=annot_mode,
                            compound_mode=compound_mode,
                            gfun_mode=gfun_mode,
                            gvar_mode=gvar_mode,
                            fsig_mode=fsig_mode)
                        tag = f'annot_mode={annot_mode}_compound_mode={compound_mode}_gfun_mode={gfun_mode}_gvar_mode={gvar_mode}_fsig_mode={fsig_mode}'

                        logger.info(f'Preprocessing {benchmark} with {tag}...')
                        for directory in tqdm.tqdm(
                                os.listdir(DATA_DIR), leave=False):
                            if os.path.isdir(
                                os.path.join(
                                    DATA_DIR,
                                    directory)):
                                OUTPUT_DIR = os.path.join(
                                    args.output_dir, DIRECTORY_NAME, tag, directory)
                                os.makedirs(OUTPUT_DIR, exist_ok=True)
                                for filename in os.listdir(
                                        os.path.join(DATA_DIR, directory)):
                                    if filename.endswith('.cas'):
                                        cass_trees = load_file(os.path.join(
                                            DATA_DIR, directory, filename), config)
                                        dense_graph = cass_tree_to_graph(
                                            cass_trees)
                                        dense_graph.save(
                                            os.path.join(
                                                OUTPUT_DIR, filename.replace(
                                                    '.cas', '.pt')))
