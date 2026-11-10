import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from tool_wear_phm2010.config.configs import ConfigLoader
from tool_wear_phm2010.ml_pipeline.tuning import Tuner
from tool_wear_phm2010.pipeline import Pipeline
from tool_wear_phm2010.utils.utils import build_parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()

    config_path = Path(os.environ['CONFIG_PATH'])
    raw_signals_dir = Path(os.environ['RAW_SIGNALS_DIR'])
    artifacts_dir = Path(os.environ['ARTIFACTS_DIR'])
    checkpoints_dir = Path(os.environ['CHECKPOINTS_DIR'])

    config = ConfigLoader.load_config(path=config_path)
    pipeline = Pipeline(
        config=config,
        data_dir=raw_signals_dir,
        artifacts_dir=artifacts_dir,
        checkpoints_dir=checkpoints_dir,
    )

    match args.mode:
        case 'train':
            # one fold: test on --cutter, train on the other labelled ones
            cutters = config.data_config.cutters
            if args.cutter not in cutters:
                raise SystemExit(f'--mode train needs --cutter in {cutters}')
            result = pipeline.run(test_cutter=args.cutter)
            logger.info(
                'Finished {cutter}: {metrics}',
                cutter=result.test_cutter,
                metrics=result.eval_result.metrics,
            )
        case 'train-all':
            # leave-one-cutter-out: every labelled cutter is tested once
            pipeline.run_all()
        case 'tune':
            # the test cutter never helps choose the hyperparameters
            Tuner(
                config=config,
                data_dir=raw_signals_dir,
                artifacts_dir=artifacts_dir,
                checkpoints_dir=checkpoints_dir,
            ).run(n_trials=args.trials)


if __name__ == '__main__':
    main()
