"""Generate policy target shaping constants from the prior deployment contract."""
import hashlib
import json
import math
from pathlib import Path


def generate(source, output):
    raw=source.read_bytes();data=json.loads(raw)
    if data['schema']!=1 or data['publication_dt_cap_s']!=0.005:
        raise ValueError('Unsupported target shaping contract')
    lines=['#pragma once','#include <array>','namespace m20::target_limits {',
           'inline constexpr char ledger_sha256[]="'+hashlib.sha256(raw).hexdigest()+'";']
    for key in ('margin_rad','max_rate_rad_s'):
        values=data[key]
        if len(values)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in values):
            raise ValueError('Invalid '+key)
        lines.append('inline constexpr std::array<double,3> '+key+'={'+','.join(repr(x) for x in values)+'};')
    lines.append('}')
    output.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();generate(a.source,a.output)
