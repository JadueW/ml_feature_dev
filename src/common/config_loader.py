import json
from pathlib import Path
"""
    负责json配置文件的加载、路径解析、输出目录创建和JSON保存
"""

def get_project_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return get_project_root() / path_obj


def ensure_dir(path_value):
    path_obj = resolve_path(path_value)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path_value):
    path_obj = resolve_path(path_value)
    with open(path_obj, 'r', encoding='utf-8') as f:
        return json.load(f)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'get_params'):
        return str(value)
    return str(value)


def save_json(path_value, payload):
    path_obj = resolve_path(path_value)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
