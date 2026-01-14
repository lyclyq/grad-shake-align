# scripts/run.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.cli import parse_args
from src.config import resolve_config
from src.hpo import run_hpo
from src.plotting import run_plotting
from src.runner import run_train
from src.final import run_final


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config, schedule_path=args.schedule, set_args=args.set)

    if args.command == "hpo":
        run_hpo(cfg, base_config_path=args.config, schedule_path=args.schedule, set_args=args.set or [])

    elif args.command == "train":
        run_train(cfg, run_id=args.run_id, trial_id=args.trial_id, trial_tag=args.trial_tag)

    elif args.command == "resume":
        cfg.setdefault("io", {})["overwrite"] = "resume"
        run_train(cfg, run_id=args.run_id, trial_id=args.trial_id, trial_tag=args.trial_tag)

    elif args.command == "plot":
        run_plotting(args.runs_dir or cfg.get("io", {}).get("root", "runs"))

    elif args.command == "final":
        run_final(cfg, best_path=args.best)

    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
