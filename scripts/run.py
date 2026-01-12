import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cli import parse_args
from src.config import resolve_config
from src.hpo import run_hpo
from src.runner import run_train


def main():
    args = parse_args()
    cfg = resolve_config(args.config, schedule_path=args.schedule, set_args=args.set)

    if args.command == 'hpo':
        run_hpo(cfg)
    elif args.command == 'train':
        run_train(cfg)
    else:
        raise ValueError(args.command)


if __name__ == '__main__':
    main()
