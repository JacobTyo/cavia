import pickle
from logger import Logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='', help='the path to the file to remove model of.')

args = parser.parse_args()

with open(args.path, 'rb') as f:
    logger = pickle.load(f)

logger.best_valid_model = None

with open(args.path, 'wb') as f:
    pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)
