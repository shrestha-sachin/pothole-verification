"""Audit first; optionally stage a new US-focused dataset. Never train or evaluate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from src.us_dataset import Audit, resolve_source, stage, verify, version_paths, write_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=ROOT/'configs/us_dataset_recipe.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Write only a new audit report (default); no staged images')
    mode.add_argument('--build', action='store_true', help='Audit, then create fresh versioned images/labels and config')
    mode.add_argument('--verify-only', action='store_true', help='Read and verify an existing output; no writes or evaluation')
    parser.add_argument('--rdd-countries', nargs='+')
    for name in ('local', 'mapillary', 'roboflow'):
        parser.add_argument('--include-'+name, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--output', type=Path, default=ROOT/'data/training/pothole_us_v2')
    parser.add_argument('--config-output', type=Path, default=ROOT/'configs/detector_us_v2.yaml')
    parser.add_argument('--report-output', type=Path, help='Must not already exist')
    parser.add_argument('--reviews', type=Path, help='Reviewed source_suitability.csv; never approval by filename alone')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--negative-ratio', type=float)
    args = parser.parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify(args.output), indent=2))
        return
    recipe = yaml.safe_load(args.recipe.read_text(encoding='utf-8'))
    # Only the new output namespace is eligible; protect all original datasets/configs.
    output = args.output.resolve()
    if output.parent != (ROOT/'data/training').resolve() or not output.name.startswith('pothole_us_'):
        parser.error('--output must be a new data/training/pothole_us_* version')
    config = args.config_output.resolve()
    if config.parent != (ROOT/'configs').resolve() or not config.name.startswith('detector_us_'):
        parser.error('--config-output must be configs/detector_us_*.yaml')
    output, config = version_paths(output, config)
    if args.report_output:
        report = args.report_output.resolve()
        if not report.is_relative_to((ROOT/'results').resolve()):
            parser.error('--report-output must be a fresh directory under results/')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report = ROOT/'results/dataset_audits'/f'us_{stamp}_{uuid.uuid4().hex[:8]}'
    if report.exists():
        parser.error('Audit report directory exists; choose a fresh --report-output')
    options = {key: getattr(args, 'include_'+key) if getattr(args, 'include_'+key) is not None else recipe.get('include_'+key, False)
               for key in ('local', 'mapillary', 'roboflow')}
    audit = Audit(ROOT, recipe, args.rdd_countries or recipe.get('rdd_countries', ['United_States']),
                  include_local=options['local'], include_mapillary=options['mapillary'], include_roboflow=options['roboflow'],
                  reviews=args.reviews, seed=args.seed if args.seed is not None else recipe.get('seed', 42),
                  negative_ratio=args.negative_ratio if args.negative_ratio is not None else recipe.get('negative_ratio', .25))
    print('Auditing source annotations, reviewed groups and protected holdout hashes...', flush=True)
    audit.run()
    write_reports(audit, report)  # Exists before staging even if a later build fails.
    console_summary = {k: v for k, v in audit.summary.items() if k != 'historical_test_pixel_hashes'}
    console_summary['protected_image_fingerprint_count'] = len(audit.summary['historical_test_pixel_hashes'])
    print(json.dumps(console_summary, indent=2))
    print(f'Audit report: {report}')
    if args.build:
        stage(audit, output, config)
        print(f'Built: {output}\nConfiguration: {config}')
    else:
        print(f'No dataset staged. Proposed output: {output}\nUse --build after inspecting the audit.')


if __name__ == '__main__':
    main()
