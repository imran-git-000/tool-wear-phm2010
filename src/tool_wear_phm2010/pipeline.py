from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

from tool_wear_phm2010.data_pipeline.data_extractor import DataExtractor
from tool_wear_phm2010.data_pipeline.data_loader import DataLoader
from tool_wear_phm2010.data_pipeline.data_transformer import DataTransformer
from tool_wear_phm2010.data_types.models import RunResult
from tool_wear_phm2010.ml_pipeline.eval_pipeline import Evaluator
from tool_wear_phm2010.models.encoder import EncoderRegressor
from tool_wear_phm2010.regressor.regressor import Regressor
from tool_wear_phm2010.utils.utils import set_seed

if TYPE_CHECKING:
    from pathlib import Path

    from tool_wear_phm2010.config.configs import Config
    from tool_wear_phm2010.data_types.models import Cut, EvalResult


class Pipeline:
    """Leave-one-cutter-out wear regression: train on the others, test one."""

    def __init__(
        self,
        config: Config,
        data_dir: Path,
        artifacts_dir: Path,
        checkpoints_dir: Path,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.checkpoints_dir = checkpoints_dir
        self.data_extractor = DataExtractor(data_dir=data_dir)

    def run(self, test_cutter: str) -> RunResult:
        """Train on every labelled cutter except test_cutter, then test it."""
        data_config = self.config.data_config
        model_config = self.config.model_config
        train_config = self.config.train_config
        train_cutters = [c for c in data_config.cutters if c != test_cutter]
        self.logger.info(
            'Running wear regression: train on {train}, test on {test}',
            train=train_cutters,
            test=test_cutter,
        )
        set_seed(train_config.seed)

        # Data Pipeline
        self.logger.info('Starting Data Pipeline')
        train_cuts, val_cuts = self._split(
            [
                cut
                for cutter in train_cutters
                for cut in self.data_extractor.iter_cuts(cutter)
            ],
        )
        test_cuts = list(self.data_extractor.iter_cuts(test_cutter))

        data_transformer = DataTransformer()
        data_transformer.fit(train_cuts)
        for cuts in (train_cuts, val_cuts, test_cuts):
            self._transform(data_transformer, cuts)

        data_loaders = DataLoader(
            window_size=model_config.window_size,
            hop=data_config.hop,
            num_workers=data_config.num_workers,
            seed=train_config.seed,
        )
        train_loader = data_loaders.get_dataloader(
            cuts=train_cuts,
            batch_size=data_config.batch_size,
            train=True,
        )
        val_loader = data_loaders.get_dataloader(
            cuts=val_cuts,
            batch_size=data_config.batch_size,
            train=False,
        )
        test_loader = data_loaders.get_dataloader(
            cuts=test_cuts,
            batch_size=data_config.batch_size,
            train=False,
        )
        self.logger.info('Terminated Data Pipeline')

        output_dir = (
            self.artifacts_dir
            / self.config.eval_config.output_artifacts
            / test_cutter
        )
        writer = SummaryWriter(log_dir=str(output_dir))

        self.logger.info('Starting training pipeline')
        regressor = Regressor(
            model=EncoderRegressor(model_config),
            train_config=train_config,
        )
        train_result = regressor.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            writer=writer,
        )
        writer.close()
        self._save_checkpoint(regressor, data_transformer, test_cutter)
        self.logger.info('Terminated training pipeline, checkpoint saved')

        self.logger.info('Starting evaluation pipeline')
        eval_result = Evaluator(
            eval_config=self.config.eval_config,
            regressor=regressor,
            test_dataloader=test_loader,
        ).evaluate()
        self._plot(
            test_cutter=test_cutter,
            cut_indices=[cut.index for cut in test_cuts],
            eval_result=eval_result,
            path=output_dir / 'predicted_vs_measured.png',
        )
        self.logger.info(
            'Terminated evaluation pipeline. Artifacts in {output}',
            output=output_dir,
        )
        return RunResult(
            test_cutter=test_cutter,
            eval_result=eval_result,
            train_result=train_result,
        )

    def run_all(self) -> list[RunResult]:
        """Every labelled cutter is the test set once."""
        results = [
            self.run(test_cutter=cutter)
            for cutter in self.config.data_config.cutters
        ]
        rmse = [r.eval_result.metrics.rmse for r in results]
        self.logger.info(
            'Leave-one-cutter-out RMSE: {mean:.4f} ± {std:.4f} mm',
            mean=np.mean(rmse),
            std=np.std(rmse),
        )
        return results

    def _split(self, cuts: list[Cut]) -> tuple[list[Cut], list[Cut]]:
        """Hold out a random share of the training cuts for validation."""
        rng = np.random.default_rng(self.config.train_config.seed)
        is_val = rng.random(len(cuts)) < self.config.data_config.val_fraction
        return (
            [cut for cut, val in zip(cuts, is_val, strict=True) if not val],
            [cut for cut, val in zip(cuts, is_val, strict=True) if val],
        )

    @staticmethod
    def _transform(data_transformer: DataTransformer, cuts: list[Cut]) -> None:
        """In place, so the raw and normalised copies never coexist."""
        for i, cut in enumerate(cuts):
            cuts[i] = data_transformer.transform(cut)

    def _save_checkpoint(
        self,
        regressor: Regressor,
        data_transformer: DataTransformer,
        test_cutter: str,
    ) -> None:
        """Weights plus the normalisation they were trained with."""
        stem = self.config.train_config.checkpoint_name
        path = self.checkpoints_dir / f'{stem}_{test_cutter}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'state_dict': regressor.state(),
                'mean': data_transformer.mean,
                'std': data_transformer.std,
            },
            path,
        )

    @staticmethod
    def _plot(
        test_cutter: str,
        cut_indices: list[int],
        eval_result: EvalResult,
        path: Path,
    ) -> None:
        """Measured vs predicted wear per cut, saved as 300 dpi PNG and PDF."""
        metrics = eval_result.metrics
        fig = Figure(figsize=(6.5, 3.0), layout='constrained')
        ax = fig.subplots()
        # mm -> µm: the dataset's own unit, easier to read
        ax.plot(
            cut_indices,
            eval_result.targets * 1000,
            color='black',
            linewidth=1.6,
            label='Measured',
        )
        ax.plot(
            cut_indices,
            eval_result.predictions * 1000,
            color='#d62728',
            linewidth=1.0,
            label='Predicted',
        )
        ax.text(
            0.02,
            0.96,
            f'RMSE = {metrics.rmse * 1000:.1f} µm\n'
            f'MAE = {metrics.mae * 1000:.1f} µm\n'
            f'$R^2$ = {metrics.r2:.2f}',
            transform=ax.transAxes,
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': '0.8', 'pad': 3},
        )
        ax.set_title(f'Test cutter {test_cutter}', fontsize=10)
        ax.set_xlabel('Cut number', fontsize=10)
        ax.set_ylabel('Flank wear (µm)', fontsize=10)
        ax.set_xlim(min(cut_indices), max(cut_indices))
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower right', fontsize=9, frameon=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix('.pdf'))
