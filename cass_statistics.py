import argparse
import logging
import os
from statistics import mean, median

import tqdm

from graph_code_embedding.cass import load_file

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
    help='The directory name that stores the data generated from download_data.py.')
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    logger.error(f'Data directory {args.data_dir} does not exist.')
    exit(1)

for benchmark in args.benchmark:
    DIRECTORY_NAME = f'Project_CodeNet_C++{benchmark}'
    DATA_DIR = os.path.join(args.data_dir, DIRECTORY_NAME, 'cass')
    if not os.path.exists(DATA_DIR):
        logger.error(
            f'Data directory {DATA_DIR} does not exist. Could not generate statistics for benchmark {benchmark}.')
        continue

    logger.info(f'Generating statistics for benchmark {benchmark}...')

    directories = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(
            os.path.join(
                DATA_DIR, d))]
    logger.info(f'Number of problems: {len(directories)}')

    all_files = []
    num_files = []
    num_cass_tree_nodes = []

    logger.info(f'Analyzing submissions...')
    for directory in tqdm.tqdm(directories, leave=False):
        files = [
            f for f in os.listdir(
                os.path.join(
                    DATA_DIR,
                    directory)) if os.path.isfile(
                os.path.join(
                    DATA_DIR,
                    directory,
                    f))]
        num_files.append(len(files))

        for file in files:
            cass_trees = load_file(os.path.join(DATA_DIR, directory, file))
            num_cass_tree_nodes.append(
                sum([len(cass_tree.nodes) for cass_tree in cass_trees]))

    logger.info(f'Number of submissions: {sum(num_files)}')

    logger.info(f'Mean number of submissions per problem: {mean(num_files)}')
    logger.info(
        f'Median number of submissions per problem: {median(num_files)}')
    logger.info(f'Min number of submissions per problem: {min(num_files)}')
    logger.info(f'Max number of submissions per problem: {max(num_files)}')

    logger.info(f'Mean number of nodes per CASS: {mean(num_cass_tree_nodes)}')
    logger.info(
        f'Median number of nodes per CASS: {median(num_cass_tree_nodes)}')
    logger.info(f'Min number of nodes per CASS: {min(num_cass_tree_nodes)}')
    logger.info(f'Max number of nodes per CASS: {max(num_cass_tree_nodes)}')
