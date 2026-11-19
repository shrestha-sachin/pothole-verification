"""Resolve project-relative YOLO roots without editing portable or historical configs."""
from pathlib import Path
import json
import uuid
import yaml


def materialize_config(path, project_root):
    path, root = Path(path).resolve(), Path(project_root).resolve()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not data.get('project_relative'):
        return path
    dataset_root = (root / data['path']).resolve()
    if not dataset_root.is_relative_to(root) or not dataset_root.is_dir():
        raise ValueError('Portable dataset root is missing or escapes project')
    if json.loads((dataset_root/'build_status.json').read_text()).get('status') != 'complete':
        raise ValueError('Dataset build is incomplete; training is disabled')
    if data.get('evaluation_role') != 'development_comparison_only' or data.get('test'):
        raise ValueError('US development datasets must not advertise a final test')
    data['path'] = dataset_root.as_posix()
    data.pop('project_relative')
    data.pop('evaluation_role')
    directory = root / 'results' / 'machine_configs'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f'{path.stem}_{uuid.uuid4().hex}.yaml'
    with output.open('x', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return output
