"""Generate the turn-onset limit; no robot or network operations."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def generate(source, output):
    raw=source.read_bytes();data=json.loads(raw);value=data['yaw_acceleration_rad_s2']
    if data['schema']!=1 or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Invalid yaw acceleration ledger')
    output.write_text('#pragma once\nnamespace m20::yaw_limits {\n'
        f'inline constexpr double acceleration={value!r};\n'
        f'inline constexpr char ledger_sha256[]="{hashlib.sha256(raw).hexdigest()}";\n'+'}\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();generate(a.source,a.output)
