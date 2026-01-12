import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Gradient Shake-to-Align Experiment Runner')
    sub = p.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', type=str, default='configs/base.yaml')
    common.add_argument('--schedule', type=str, default=None, help='Optional schedule yaml to merge on top of base')
    common.add_argument('--set', action='append', default=None, help='Override keys: --set train.lr=3e-5')

    # Runtime IO controls
    common.add_argument('--run_id', type=str, default=None, help='Reuse a specific run id (advanced)')

    sub.add_parser('train', parents=[common], help='Single run (one config, one seed)')
    sub.add_parser('hpo', parents=[common], help='Run hyperparameter search')
    sub.add_parser('final', parents=[common], help='Run final evaluation suite using best hpo params')

    return p.parse_args()
