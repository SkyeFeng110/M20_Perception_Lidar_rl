"""Hash and guarded-edit utilities; no network or robot operations."""
import hashlib
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Missing or ambiguous source anchor: '+before[:90])
    return text.replace(before, after)


def refresh_checksums(release):
    entries=[]
    for path in sorted(release.rglob('*')):
        relative=path.relative_to(release)
        if relative.parts[0] in ('build','log','artifacts','.runtime','incidents') or '__pycache__' in relative.parts:
            continue
        if path.name == '.m20_gos_native.lock':
            continue  # Ephemeral supervisor lock, not a release input.
        if path.is_file() and not path.is_symlink() and path.name!='RELEASE_SHA256SUMS':
            entries.append(digest(path)+'  '+str(relative)+'\n')
    (release/'RELEASE_SHA256SUMS').write_text(''.join(entries))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('release',type=Path)
    refresh_checksums(parser.parse_args().release.resolve())
