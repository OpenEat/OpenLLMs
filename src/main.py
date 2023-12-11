import os
import sys
import argparse

sys.path.append("../")
from utils.reader import read_yaml
from modules.controller import Controller

def main(args):
    """ main """
    config = read_yaml(args.pipeline_config)
    controller = Controller(config)
    controller.register()
    controller.dispatch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config", type=str, help="pipeline config of the experiment")
    args = parser.parse_args()
    main(args)