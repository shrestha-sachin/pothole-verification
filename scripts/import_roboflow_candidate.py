"""Copy a local YOLO export into a NEW candidate source, never into training.

For authenticated access, export YOLO detection data from Roboflow manually,
or use --download with the optional roboflow SDK and ROBOFLOW_API_KEY.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.us_dataset import roboflow_metadata, parse_yolo, IMAGE_SUFFIXES


def validate_export(directory):
    directory = Path(directory).resolve()
    report = roboflow_metadata(directory.parent, directory)
    if not report.get('available'):
        raise ValueError('Export must contain data.yaml')
    from PIL import Image
    for image in directory.glob('*/images/*'):
        if image.suffix.lower() in IMAGE_SUFFIXES:
            label = image.parent.parent/'labels'/f'{image.stem}.txt'
            parse_yolo(label.read_text(encoding='utf-8'), report['classes'])
            with Image.open(image) as im:
                im.verify()
    for label in directory.glob('*/labels/*.txt'):
        if not any((label.parent.parent/'images'/f'{label.stem}{ext}').is_file() for ext in IMAGE_SUFFIXES):
            raise ValueError('Label has no matching image')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source', type=Path, help='Existing local YOLO detection export containing data.yaml')
    mode.add_argument('--download', action='store_true', help='Explicit optional network download; never automatic')
    parser.add_argument('--output', type=Path, required=True, help='Fresh data/raw/roboflow_candidates/<version> directory')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'data/raw/roboflow_candidates').resolve()) or output == (ROOT/'data/raw/roboflow_candidates').resolve():
        parser.error('Output must be a new version below data/raw/roboflow_candidates/')
    if output.exists():
        parser.error('Output exists; choose a new version')
    if args.source:
        source = args.source.resolve()
        validate_export(source)  # Validate schema and labels before copying anything.
        if not (source/'data.yaml').is_file():
            parser.error('Source must contain data.yaml')
        shutil.copytree(source, output)
    else:
        key = os.environ.get('ROBOFLOW_API_KEY')
        if not key:
            parser.error('Set ROBOFLOW_API_KEY in the environment; never put credentials in source files')
        # SDK/network support is deliberately outside core preparation dependencies.
        try:
            from roboflow import Roboflow
            # Third-party SDKs may log authenticated export URLs. Suppress logs and
            # do not expose exception bodies, which may also contain credentials.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rf = Roboflow(api_key=key)
                rf.workspace('the-american-university-of-the-middle-east').project('pothole-detection-3uuq2').version(2).download('yolov11', location=str(output), overwrite=False)
        except Exception:
            raise SystemExit('Optional download failed. Check SDK installation/access privately, or use a manual local export.') from None
    validate_export(output)
    metadata = roboflow_metadata(ROOT, output)
    with (output/'candidate_review.json').open('x', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Candidate stored at {output}. Status: candidate_not_approved. No training recipe changed.')


if __name__ == '__main__':
    main()
