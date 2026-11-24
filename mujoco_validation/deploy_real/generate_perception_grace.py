"""Generate the sole runtime limit from the operator-confirmed loss ledger."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def generate(source, output):
    raw = source.read_bytes()
    data = json.loads(raw)
    value = data['loss_timeout_s']
    if (data.get('schema') != 1 or data.get('operator_confirmed') is not True or
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise ValueError('Invalid or unconfirmed perception grace ledger')
    output.write_text('#pragma once\nnamespace m20::perception_limits {\n'
                      f'inline constexpr double loss_timeout_s={value!r};\n'
                      f'inline constexpr char ledger_sha256[]="{hashlib.sha256(raw).hexdigest()}";\n'
                      '}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    generate(args.source, args.output)
