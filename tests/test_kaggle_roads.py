from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.prepare_kaggle_roads import pothole_labels, prepare


@pytest.mark.parametrize('line', ['0 nan .5 .2 .2', '0 .9 .5 0 .2', '3 .5 .5 .2 .2', '0 .5 .5'])
def test_reject_invalid_labels(line):
    with pytest.raises(ValueError):
        pothole_labels(line)


def test_filter_classes():
    assert pothole_labels('0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n2 .5 .5 .2 .2') == ['0 0.500000 0.500000 0.200000 0.200000']


def test_clip_source_boxes_at_boundary():
    assert pothole_labels('0 .9 .5 .4 .2') == ['0 0.850000 0.500000 0.300000 0.200000']
    assert pothole_labels('1 .5 .5 .2 0') == []


def test_preserve_splits_and_exclude_existing_pixels(tmp_path):
    source = tmp_path / 'source'
    (source / 'images').mkdir(parents=True)
    (source / 'labels-YOLO').mkdir()
    for name, color, label in [('a', 'red', '0 .5 .5 .2 .2'), ('b', 'blue', '1 .5 .5 .2 .2'), ('c', 'green', '0 .5 .5 .2 .2')]:
        Image.new('RGB', (20, 20), color).save(source / 'images' / f'{name}.png')
        (source / 'labels-YOLO' / f'{name}.txt').write_text(label)
    existing = tmp_path / 'val.txt'
    existing.write_text((source / 'images/a.png').as_posix() + '\n')
    train = tmp_path / 'train.txt'
    train.write_text('')
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump({'path': str(tmp_path), 'train': str(train), 'val': str(existing)}))
    output, config = tmp_path / 'output', tmp_path / 'combined.yaml'
    summary = prepare(source, output, base, config)
    assert summary['duplicates_excluded'] == 1
    assert summary['positive_images'] == summary['negative_images'] == 1
    combined = yaml.safe_load(config.read_text())
    assert combined['val'] == [existing.as_posix()]
    assert len(combined['train']) == 2
    assert (output / 'labels/arcioni_00000.txt').read_text() == ''
    assert existing.read_text() == (source / 'images/a.png').as_posix() + '\n'
    with pytest.raises(FileExistsError):
        prepare(source, output, base, config)
