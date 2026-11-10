from dataclasses import fields, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from tool_wear_phm2010.pipeline import Pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from tool_wear_phm2010.config.configs import Config
    from tool_wear_phm2010.data_types.models import RunResult

Params = dict[str, float | int]

# each name is a field of exactly one sub-config of Config
SEARCH_SPACE: dict[str, list[float | int]] = {
    'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
    'weight_decay': [0.0, 1e-4, 1e-3, 1e-2],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'batch_size': [64, 128, 256],
    'hidden_size': [64, 128, 256],
    'num_hidden_layers': [1, 2, 3, 4],
    'num_attention_heads': [4, 8],
    'intermediate_size': [256, 512],
    'window_size': [128, 256, 512, 1024, 2048, 4096],
    'patch_size': [8, 16, 32, 64, 128, 256],
}
# tokens per window: fewer leaves attention nothing to relate, more is slow
MIN_TOKENS, MAX_TOKENS = 4, 256


def sample_params(n: int, seed: int) -> list[Params]:
    """n uniform draws from SEARCH_SPACE with a valid window/patch pair."""
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        params = {
            name: values[rng.integers(len(values))]
            for name, values in SEARCH_SPACE.items()
        }
        tokens = params['window_size'] // params['patch_size']
        if MIN_TOKENS <= tokens <= MAX_TOKENS:
            samples.append(params)
    return samples


def apply_params(base: Config, params: Params) -> Config:
    """Write each param into the sub-config that owns it; hop = window."""
    sections = {}
    for section in fields(base):
        sub_config = getattr(base, section.name)
        owned = {f.name for f in fields(sub_config)}
        sections[section.name] = replace(
            sub_config,
            **{k: v for k, v in params.items() if k in owned},
        )
    config = replace(base, **sections)
    return replace(
        config,
        data_config=replace(
            config.data_config,
            hop=config.model_config.window_size,
        ),
    )


class Tuner:
    """Nested leave-one-cutter-out random search.

    For each test cutter, candidates are scored on the other two cutters
    only (train on one, test on the other, both ways). The best one is then
    trained on both and tested on the held-out cutter, which never helps
    choose the hyperparameters.
    """

    def __init__(
        self,
        config: Config,
        data_dir: Path,
        artifacts_dir: Path,
        checkpoints_dir: Path,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.config = config
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.checkpoints_dir = checkpoints_dir
        self.tuning_dir = artifacts_dir / 'tuning'

    def run(self, n_trials: int) -> list[RunResult]:
        # the same candidates for every test cutter
        candidates = sample_params(
            n_trials, seed=self.config.train_config.seed
        )
        cutters = self.config.data_config.cutters
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        results: list[RunResult] = []
        for test_cutter in cutters:
            inner_cutters = [c for c in cutters if c != test_cutter]
            scores = []
            for trial, params in enumerate(candidates):
                inner_rmse = self._inner_rmse(
                    params=params,
                    inner_cutters=inner_cutters,
                    test_cutter=test_cutter,
                    trial=trial,
                )
                score = float(np.mean(list(inner_rmse.values())))
                scores.append(score)
                rows.append(
                    {
                        'test_cutter': test_cutter,
                        'trial': trial,
                        **params,
                        **{f'rmse_{c}': v for c, v in inner_rmse.items()},
                        'score': score,
                    },
                )
                # rewritten every trial so a long run can be inspected live
                pd.DataFrame(rows).to_csv(
                    self.tuning_dir / 'results.csv',
                    index=False,
                )

            best = int(np.argmin(scores))
            self.logger.info(
                'Best for {test}: trial {trial}, inner RMSE {score:.4f}, '
                '{params}',
                test=test_cutter,
                trial=best,
                score=scores[best],
                params=candidates[best],
            )
            results.append(self._final_run(candidates[best], test_cutter))

        rmse = [r.eval_result.metrics.rmse for r in results]
        self.logger.info(
            'Nested LOCO RMSE: {per_cutter}, mean {mean:.4f} ± {std:.4f} mm',
            per_cutter={
                r.test_cutter: round(v, 4)
                for r, v in zip(results, rmse, strict=True)
            },
            mean=np.mean(rmse),
            std=np.std(rmse),
        )
        return results

    def _inner_rmse(
        self,
        params: Params,
        inner_cutters: list[str],
        test_cutter: str,
        trial: int,
    ) -> dict[str, float]:
        """RMSE on each inner cutter when trained on the other one."""
        config = apply_params(self.config, params)
        name = f'trial_{trial:02d}'
        config = replace(
            config,
            data_config=replace(config.data_config, cutters=inner_cutters),
            train_config=replace(config.train_config, checkpoint_name=name),
            eval_config=replace(config.eval_config, output_artifacts=name),
        )
        fold_dir = self.tuning_dir / test_cutter
        pipeline = Pipeline(
            config=config,
            data_dir=self.data_dir,
            artifacts_dir=fold_dir,
            checkpoints_dir=fold_dir / 'checkpoints',
        )
        try:
            return {
                r.test_cutter: r.eval_result.metrics.rmse
                for r in pipeline.run_all()
            }
        except RuntimeError:
            # divergence or CUDA out of memory: a bad config, not a crash
            self.logger.exception('Trial {trial} failed', trial=trial)
            return {c: float('inf') for c in inner_cutters}

    def _final_run(self, params: Params, test_cutter: str) -> RunResult:
        """Train on both inner cutters with the chosen params, test once."""
        config = apply_params(self.config, params)
        config = replace(
            config,
            train_config=replace(config.train_config, checkpoint_name='tuned'),
            eval_config=replace(config.eval_config, output_artifacts='tuned'),
        )
        return Pipeline(
            config=config,
            data_dir=self.data_dir,
            artifacts_dir=self.artifacts_dir,
            checkpoints_dir=self.checkpoints_dir,
        ).run(test_cutter=test_cutter)
