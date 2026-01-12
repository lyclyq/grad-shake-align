import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cli import parse_args
from src.config import resolve_config
from src.hpo import run_hpo
from src.plotting import run_plotting
from src.runner import run_train


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config, schedule_path=args.schedule, set_args=args.set)

    if args.command == 'hpo':
        run_hpo(cfg, base_config_path=args.config, schedule_path=args.schedule, set_args=args.set or [])
    elif args.command == 'train':
        run_train(cfg, run_id=args.run_id, trial_id=args.trial_id)
    elif args.command == 'resume':
        cfg.setdefault('io', {})['overwrite'] = 'resume'
        run_train(cfg, run_id=args.run_id, trial_id=args.trial_id)
    elif args.command == 'plot':
        run_plotting(args.runs_dir or cfg.get('io', {}).get('root', 'runs'))
    else:
        raise ValueError(args.command)


if __name__ == '__main__':
    main()
