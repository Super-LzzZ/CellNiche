'''
Author: Zhongming Liang lzzzmmgpt@gmail.com
Date: 2025-06-20 22:52:59
LastEditors: Zhongming Liang lzzzmmgpt@gmail.com
LastEditTime: 2025-06-21 17:10:52
FilePath: /530/CellNiche/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# main.py

import argparse
import logging
import yaml

from trainer import run

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entry point for CellNiche training & inference"
    )
    # — only one extra flag, everything else comes from config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()

def main():
    # 1) Load config file
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2) Turn config keys into attributes of a simple args object
    class Obj: pass
    opts = Obj()
    for k, v in cfg.items():
        setattr(opts, k, v)

    # 3) Set up logging
    level = logging.INFO if opts.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 4) Kick off training/inference
    run(opts)

if __name__ == "__main__":
    main()
