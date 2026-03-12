import re
from pathlib import Path


FILENAME_PATTERN = re.compile(r'(?P<subject>[^-]+)-(?P<implant>[^-]+)-(?P<task_type>[^-]+)-day(?P<day>\d+)-(?P<recording>\d+)\.pkl$')


def discover_recordings(raw_dir, subject=None, task_type=None):
    """
    从raw_dir中获取pkl文件，并解析 path/subject / implant / task_type / day / recording等字段
    :param raw_dir:
    :param subject:
    :param task_type:
    :return:
    """
    raw_dir = Path(raw_dir)
    recordings = []
    for path_obj in sorted(raw_dir.glob('*.pkl')):
        match = FILENAME_PATTERN.match(path_obj.name)
        if not match:
            continue
        info = {
            'path': path_obj,
            'subject': match.group('subject'),
            'implant': match.group('implant'),
            'task_type': match.group('task_type'),
            'day': int(match.group('day')),
            'recording': int(match.group('recording')),
            'recording_id': 'day%s-rec%s' % (match.group('day'), match.group('recording'))
        }
        if subject is not None and info['subject'] != subject:
            continue
        if task_type is not None and info['task_type'] != task_type:
            continue
        recordings.append(info)
    return recordings


def group_by_day(recordings):
    grouped = {}
    for info in recordings:
        key = 'day%s' % info['day']
        grouped.setdefault(key, []).append(info)
    return grouped
