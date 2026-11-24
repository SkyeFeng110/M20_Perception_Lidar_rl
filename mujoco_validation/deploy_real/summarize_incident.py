"""Read the bounded onboard recorder without changing files or controlling motors."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(directory):
    sessions = {}
    errors = []
    files = {}
    for path in sorted(directory.glob('segment_*.jsonl')):
        raw = path.read_bytes()
        files[path.name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        session = None
        for line_no, line in enumerate(raw.splitlines(), 1):
            try:
                item = json.loads(line)
            except (ValueError, UnicodeError):
                errors.append({'file': path.name, 'line': line_no, 'reason': 'malformed_or_partial_row'})
                continue
            if item.get('event') == 'segment_start':
                session = item['session']
                s = sessions.setdefault(session, {'header': item, 'records': [], 'ended': False})
                if item['wall_s'] > s['header']['wall_s']:
                    s['header'] = item
            elif session is None:
                errors.append({'file': path.name, 'line': line_no, 'reason': 'missing_segment_header'})
            elif item.get('session') != session:
                errors.append({'file': path.name, 'line': line_no, 'reason': 'session_mismatch'})
            elif item.get('event') == 'session_end':
                sessions[session]['ended'] = True
                sessions[session]['footer'] = item
            elif item.get('event') == 'incident_frozen':
                sessions[session]['frozen'] = item
            else:
                sessions[session]['records'].append(item)
    results = []
    for session, s in sessions.items():
        records = sorted(s['records'], key=lambda x: x['record_seq'])
        gaps = []
        for prev, current in zip(records, records[1:]):
            if current['record_seq'] != prev['record_seq'] + 1:
                gaps.append([prev['record_seq'], current['record_seq']])
        first = next((r for r in records if r.get('joint_fault_latched') and
                      r.get('reason') not in ('joint_fault_latched', 'rejected_published_damping')), None)
        results.append({'session': session, 'model_sha256': s['header']['model_sha256'],
                        'ledger_sha256': s['header']['ledger_sha256'], 'retained_records': len(records),
                        'retained_first_seq': records[0]['record_seq'] if records else None,
                        'older_history_overwritten': bool(records and records[0]['record_seq'] > 0),
                        'sequence_gaps': gaps, 'session_end_seen': s['ended'],
                        'footer': s.get('footer'), 'frozen': s.get('frozen'), 'first_retained_joint_fault': first})
    first_fault = None
    path=directory/'first_fault.json'
    if path.exists():
        try:first_fault=json.loads(path.read_bytes())
        except (ValueError, UnicodeError):errors.append({'file':path.name,'reason':'malformed_first_fault'})
    return {'files': files, 'parse_errors': errors, 'sessions': results, 'saved_first_fault':first_fault,
            'scope': 'retained ring only; an absent end may mean running, crash or power loss; not accident root-cause proof'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    print(json.dumps(summarize(a.directory), indent=2, ensure_ascii=False))
