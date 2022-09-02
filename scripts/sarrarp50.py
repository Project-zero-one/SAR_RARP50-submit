import argparse
import sys
from scripts.evaluate import main as main_e
from scripts.generate_predictions import main as main_g


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Subcommand to run', choices=['evaluate', 'generate', 'unpack'])
    cmd_tool = parser.parse_args(sys.argv[1:2])
    print (cmd_tool.command)
    if cmd_tool.command == 'evaluate':
        parser =argparse.ArgumentParser()
        parser.add_argument('ref_dir')
        parser.add_argument('prediction_dir')
        parser.add_argument( '--ignore_actions', action='store_true')
        parser.add_argument( '--ignore_segmentation', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        if (args.ignore_actions and args.ignore_segmentation):
            parser.error('--ignore_actions and --ignore_segmentation flags cannot be set at the same time')
        SystemExit(main_e(args))

    elif cmd_tool.command == 'generate': 
        parser = argparse.ArgumentParser()
        parser.add_argument('test_dir', help='root_directory of the test set')
        parser.add_argument('prediction_dir', help='directory to store predictions')
        parser.add_argument('task', type=str, help='segmentation or actions or multitask')
        parser.add_argument('-o','--overwrite', help='overwrite the predictions of prediction_dir', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        SystemExit(main_g(args))
    else:
        raise NotImplementedError
