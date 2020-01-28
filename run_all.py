import os
import argparse

parser = argparse.ArgumentParser(description='Classification tests for UCR repository datasets')
parser.add_argument('--path',action='store_true', default="../UCRArchive_2018/", help="Enter the path to master folder containing all the tests!")
args = parser.parse_args()

for files in os.listdir(args.path):
	os.system(' python ucr.py --dataset {} --path ../UCRArchive_2018/ --save_path models/ --hyper default_hyperparameters.json --cuda --gpu 0 '.format(files))
