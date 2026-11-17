import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import prepare_training_dataset as builder


def test_negative_budget_preserves_positives_and_strata():
    rows = [dict(source='a', boxes=['box'], stratum='pos', id=i) for i in range(8)]
    rows += [dict(source='a', boxes=[], stratum=str(i % 2), id=i+8) for i in range(20)]
    result = builder.sample_negatives(rows, 0.25, 42)
    assert sum(bool(r['boxes']) for r in result) == 8
    assert {r['stratum'] for r in result if not r['boxes']} == {'0', '1'}
    assert result == builder.sample_negatives(rows, 0.25, 42)


def test_staging_protects_holdouts_and_pairs_labels(tmp_path, monkeypatch):
    rows = []
    for i, (split, color) in enumerate([('train', 'red'), ('train', 'blue'), ('val', 'green'), ('test', 'yellow'), ('train', 'yellow')]):
        image, label = tmp_path / f'{i}.png', tmp_path / f'{i}.txt'
        Image.new('RGB', (20, 20), color).save(image)
        label.write_text('0 .5 .5 .2 .2\n')
        rows.append(dict(image=image, label=label, split=split, group=str(i), source='fixture', stratum='positive', original_id=str(i)))
    monkeypatch.setattr(builder, 'collect', lambda: rows)
    out, config = tmp_path / 'out', tmp_path / 'config.yaml'
    summary = builder.prepare(out, config)
    assert summary['excluded']['exact_duplicate'] == 1
    assert summary['splits']['train']['images'] == 2
    for split in ('train', 'val', 'test'):
        for line in (out / f'{split}.txt').read_text().splitlines():
            image = Path(line)
            assert image.exists()
            assert (out / 'labels' / split / (image.stem+'.txt')).read_text().strip()
    with pytest.raises(FileExistsError):
        builder.prepare(out, config)


def test_group_overlap_fails_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'collect', lambda: [dict(group='recording', split=s) for s in ('train', 'test')])
    with pytest.raises(ValueError, match='split overlap'):
        builder.prepare(tmp_path / 'out', tmp_path / 'config.yaml')
    assert not (tmp_path / 'out').exists()


def test_non_pothole_labels_are_rejected():
    with pytest.raises(ValueError, match='class 0'):
        builder.validate_labels('1 .5 .5 .2 .2')
