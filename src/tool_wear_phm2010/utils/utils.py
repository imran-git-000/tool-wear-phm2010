import argparse
import random

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='tool_wear_phm2010',
        description='Train and evaluate a tool wear regressor on PHM 2010.',
    )
    parser.add_argument(
        '--mode',
        choices=('train', 'train-all'),
        default='train',
        help='train: train on the other labelled cutters, test on --cutter. '
        'train-all: leave-one-cutter-out over every labelled cutter. '
        '(default: train)',
    )
    parser.add_argument(
        '--cutter',
        default=None,
        help='test cutter for train, one of the labelled cutters in the '
        'config (e.g. c4)',
    )
    return parser


def resolve_device(device: str) -> torch.device:
    """Turn the configured device string into a torch device."""
    if device != 'auto':
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int) -> None:
    """Seed python and torch so a run can be reproduced.

    Call this *before* building the model: a module initialises its weights
    in its constructor, so seeding afterwards leaves them unreproducible.
    torch.manual_seed covers the CUDA devices as well. Nothing here draws
    from numpy's global RNG; use np.random.default_rng(seed) if that changes.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
