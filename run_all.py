import os
import argparse

def main(path):
	for files in os.listdir(path, cuda):
		if cuda:
			exec(' python ucr.py --dataset {} --path ../UCRArchive_2018/ --save_path models/ --hyper default_hyperparameters.json --cuda --gpu 0 --sliding_window 1'.format(files))
		else:
			exec(' python ucr.py --dataset {} --path ../UCRArchive_2018/ --save_path models/ --hyper default_hyperparameters.json --sliding_window 1'.format(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--cuda',action='store_true', type=int, default=0
    					help="if wanna use cuda then enter 1")
    parser.add_argument('--path',action='store_true', required=True, 
    					default="../UCRArchive_2018/", help="Enter the path to master folder containing all the tests!")
    parser.parse_args()
	main(args.path)