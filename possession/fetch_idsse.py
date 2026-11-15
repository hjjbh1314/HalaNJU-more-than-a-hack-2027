import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CATALOG = 'https://api.figshare.com/v2/articles/28196177/versions/1'


def download(entry, directory):
    path = directory / entry['name']
    source = path if path.is_file() else path.with_suffix('.xml.part')
    if source != path:
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '3', '--output', str(source), entry['download_url']], check=True)
    with source.open('rb') as handle:
        checksum = hashlib.file_digest(handle, 'md5').hexdigest()
    if source.stat().st_size != entry['size'] or checksum != entry['computed_md5']:
        raise ValueError(f'{source.name} 与 Figshare 发布的长度或 MD5 不符')
    if source != path:
        source.replace(path)
    print(f'{path.name}  {path.stat().st_size:,} bytes  MD5 OK', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--events-only', action='store_true', help='下载七场外部比赛的事件表，供换队诊断使用')
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=True, parents=True)
    catalog = json.loads(subprocess.check_output(['curl', '--fail', '--location', '--silent',
                                                  '--show-error', CATALOG]))
    prefixes = ('DFL_03_02_',) if args.events_only else ('DFL_02_01_', 'DFL_04_03_')
    entries = sorted((item for item in catalog['files'] if item['name'].startswith(prefixes)),
                     key=lambda item: item['name'])
    if len(entries) != 7 * len(prefixes) or catalog['version'] != 1:
        raise ValueError('IDSSE 第 1 版比赛清单不完整')
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda entry: download(entry, args.directory), entries))


if __name__ == '__main__':
    main()
