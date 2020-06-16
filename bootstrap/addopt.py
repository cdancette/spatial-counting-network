from bootstrap import Options
import yaml
import argparse
import shutil
import os
"""
This script will modify or add any option you want in existing yaml files.
For example:
    
    python -m bootstrap.addopt -o logs/resnet152/options.yaml \
    -a model.network.share_params true \
    -a model.criterion.entropy_loss

Values will be parsed with yaml. A missing value will be set to None

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--options", nargs="+")
    parser.add_argument("-a", nargs="+", action="append")
    args = parser.parse_args()
    print(args.a)
    for path in args.options:
        if os.path.exists(path):
            shutil.copy(path, path+".bak")  # backup
            print(f"Processing {path}")
            options = Options.load_yaml_opts(path)
            for opt in args.a:
                if len(opt) == 1:
                    options[opt[0]] = None
                else:
                    options[opt[0]] = yaml.safe_load(opt[1])
            Options.save_yaml_opts(options, path)


if __name__ == "__main__":
    main()
