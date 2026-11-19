"""Reproduce the archived US V4 comparison without training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from src.us_dataset import write_reports
from src.us_dataset_v4 import V4Audit, reports_v4, stage_v4, verify_v4


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=ROOT/'configs/us_dataset_v4_recipe.yaml')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--build',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'data/training/pothole_us_v4')
    p.add_argument('--config-output',type=Path,default=ROOT/'configs/detector_us_v4.yaml')
    p.add_argument('--report-output',type=Path)
    p.add_argument('--review-dir',type=Path,default=ROOT/'results/dataset_reviews/v4')
    p.add_argument('--intake-manifest',type=Path)
    p.add_argument('--recording-assignments',type=Path)
    p.add_argument('--rdd-countries',nargs='+')
    negatives=p.add_mutually_exclusive_group()
    negatives.add_argument('--negative-ratio',type=float)
    negatives.add_argument('--keep-all-reviewed-negatives',action='store_true')
    p.add_argument('--seed',type=int)
    for name in ('local','mapillary','roboflow'):
        p.add_argument('--include-'+name,action=argparse.BooleanOptionalAction,default=None)
    args=p.parse_args(argv)
    if args.verify_only:
        output = args.output.resolve()
        if output == ROOT/'data/training/pothole_us_v4' and not output.exists():
            output = ROOT/'results/dataset_archive/retired_us_builds/data/training/pothole_us_v4'
        print(json.dumps(verify_v4(output),indent=2));return
    output,config=args.output.resolve(),args.config_output.resolve()
    if output.parent!=(ROOT/'data/training').resolve() or not output.name.startswith('pothole_us_') or output.name in {'pothole_us_v2','pothole_us_v3'}:
        p.error('Choose a fresh pothole_us_* output in data/training; V2/V3 are protected')
    if config.parent!=(ROOT/'configs').resolve() or not config.name.startswith('detector_us_') or config.name in {'detector_us_v2.yaml','detector_us_v3.yaml'}:
        p.error('Choose a fresh configs/detector_us_*.yaml; V2/V3 are protected')
    if args.build and (output.exists() or config.exists()):
        p.error('Output/config exists. Choose explicit new comparison names; nothing will be overwritten.')
    report=args.report_output.resolve() if args.report_output else ROOT/'results/dataset_audits'/('v4_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    if not report.is_relative_to((ROOT/'results').resolve()) or report.exists():
        p.error('Choose a fresh report directory below results/')
    recipe=yaml.safe_load(args.recipe.read_text(encoding='utf-8'))
    opts={name:getattr(args,'include_'+name) if getattr(args,'include_'+name) is not None else recipe.get('include_'+name,False)
          for name in ('local','mapillary','roboflow')}
    audit=V4Audit(ROOT,recipe,args.rdd_countries or recipe['rdd_countries'],
        review_dir=args.review_dir,intake=args.intake_manifest,assignments=args.recording_assignments,
        include_local=opts['local'],include_mapillary=opts['mapillary'],include_roboflow=opts['roboflow'],
        negative_ratio=None if args.keep_all_reviewed_negatives else args.negative_ratio if args.negative_ratio is not None else recipe['negative_ratio'],
        seed=args.seed if args.seed is not None else recipe['seed'])
    print('Auditing original labels, V3 exclusions, every eligible negative, and the frozen validation reference...',flush=True)
    audit.run()
    write_reports(audit,report);reports_v4(audit,report)
    print(json.dumps({k:v for k,v in audit.summary.items() if k!='historical_test_pixel_hashes'},indent=2))
    print(f'Audit and human-review queues: {report}')
    if args.build:
        stage_v4(audit,output,config)
        print(f'Built: {output}\nConfig: {config}')
    else:print('Audit only. No images, annotations, review decisions, or existing datasets changed.')


if __name__=='__main__':main()
