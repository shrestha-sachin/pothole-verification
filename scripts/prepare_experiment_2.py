"""Audit, build or verify the pothole-only Experiment 2 dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_2 import Experiment2Audit, stage_experiment, verify_experiment, write_report
from src.us_dataset import version_paths


def ratio(value):
    return None if value.lower() == 'all' else float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Audit and write review queues only (default)')
    mode.add_argument('--build', action='store_true', help='Audit, then stage a fresh dataset and verify it')
    mode.add_argument('--verify-only', action='store_true', help='Verify an existing build without writing to it')
    parser.add_argument('--negative-ratio', type=ratio, default=argparse.SUPPRESS, help='Negatives per positive (default 1.0), or all')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, help='Build destination or existing dataset to verify')
    parser.add_argument('--config-output', type=Path, default=ROOT/'configs/detector_experiment_2.yaml')
    parser.add_argument('--review-report', type=Path, help='Fresh audit directory below results/')
    parser.add_argument('--review-dir', type=Path, default=ROOT/'results/dataset_reviews/experiment_2')
    parser.add_argument('--recipe', type=Path, default=ROOT/'configs/experiment_2_recipe.yaml')
    parser.add_argument('--exclude-roboflow', action='store_true', help='Omit Roboflow even if image reviews pass')
    parser.add_argument('--exclude-rdd-negatives', action='store_true', help='Use only local/Mapillary and reviewed Roboflow negatives')
    args = parser.parse_args(argv)
    try:
        output = args.output or ROOT/'data/training/pothole_experiment_2'
        if args.verify_only:
            if args.output is None and args.config_output.is_file():
                config = yaml.safe_load(args.config_output.read_text(encoding='utf-8'))
                output = ROOT/config['path']
            print(json.dumps(verify_experiment(output), indent=2))
            return
        output, config = output.resolve(), args.config_output.resolve()
        if output.parent != (ROOT/'data/training').resolve() or not re.fullmatch(r'pothole_experiment_2(?:_[a-zA-Z0-9_-]+)?', output.name):
            raise ValueError('Choose an Experiment 2 output under data/training/')
        if config.parent != (ROOT/'configs').resolve() or not re.fullmatch(r'detector_experiment_2(?:_[a-zA-Z0-9_-]+)?\.yaml', config.name):
            raise ValueError('Choose an Experiment 2 YAML under configs/')
        if args.build:
            new_output, new_config = version_paths(output, config)
            if (new_output, new_config) != (output, config):
                print(f'Existing Experiment 2 artifacts are reserved. Using {new_output.name} and {new_config.name}.')
            output, config = new_output, new_config
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report = args.review_report.resolve() if args.review_report else ROOT/'results/dataset_audits'/f'experiment_2_{stamp}_{uuid.uuid4().hex[:8]}'
        if not report.is_relative_to((ROOT/'results').resolve()) or report.exists():
            raise ValueError('Choose a fresh review report directory below results/')
        recipe = yaml.safe_load(args.recipe.read_text(encoding='utf-8'))
        audit = Experiment2Audit(ROOT, recipe, review_dir=args.review_dir,
            negative_ratio=getattr(args, 'negative_ratio', recipe['negative_ratio']),
            seed=args.seed if args.seed is not None else recipe['seed'],
            include_roboflow=not args.exclude_roboflow, include_rdd_negatives=not args.exclude_rdd_negatives)
        print('Checking Japan and Norway XML, reviewed local frames, Roboflow, and protected split fingerprints.', flush=True)
        audit.run()
        write_report(audit, report)
        print(json.dumps(dict(splits=audit.summary['splits'], exclusions=audit.summary['excludes'],
                              validation_fingerprint=audit.summary['validation_fingerprint']), indent=2))
        print(f'Audit and review queues: {report}')
        if args.build:
            stage_experiment(audit, output, config)
            print(f'Verified dataset: {output}\nPortable configuration: {config}')
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
