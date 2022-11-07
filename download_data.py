import argparse
import logging
import os
import requests
import shutil
import tarfile

import tqdm


logging.basicConfig(format='[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='The directory in which to download the data.')
parser.add_argument('--benchmark', type=int, default=1000, nargs='+', choices=[1000, 1400], help='The benchmark number.')
parser.add_argument('--datatype', type=str, default='code', nargs='+', choices=['code', 'spts', 'cass'], help='The type of data to download. SPT = simplified parse tree, CASS = context-aware semantics structure.')
args = parser.parse_args()

# Create or clear the data directory
if os.path.exists(args.data_dir):
    logger.info(f'Clearing data directory...')
    shutil.rmtree(args.data_dir)
os.makedirs(args.data_dir)

# Download tar archive to local disk
DATA_ENDPOINT = f'https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/'
get_filename = lambda benchmark, datatype: f'Project_CodeNet_C++{benchmark}{"" if datatype == "code" else "_" + datatype}.tar.gz'
for benchmark in args.benchmark:
    for datatype in args.datatype:
        file_name = get_filename(benchmark, datatype)
        url = DATA_ENDPOINT + file_name
        file_path = os.path.join(args.data_dir, file_name)

        logger.info(f'Downloading {url} to {file_path}...')
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            file_size = int(r.headers.get('Content-Length', 0))
            progress_bar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(1024)
            progress_bar.close()
            logger.info(f'Downloaded {url} to {file_path}.')
            logger.info(f'Extracting {file_name}...')
            with tarfile.open(file_path) as tf:
                for member in tqdm.tqdm(iterable=tf.getmembers(), total=len(tf.getmembers()), leave=False):
                    tf.extract(member, path=args.data_dir)
            logger.info(f'Extracted {file_name}.')
            os.remove(file_path)
        else:
            logger.error(f'Error downloading {url}.')
