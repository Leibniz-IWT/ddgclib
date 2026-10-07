"""``python -m ddgclib.methods``: print the method registry / presets."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ddgclib.methods import (
    AXES, PRESETS, axes_markdown, config_markdown, presets_markdown, preset,
)


def update_methods_md(path: str | Path) -> bool:
    """Rewrite the generated sections 2 (axes) and 3 (presets) of
    ``METHODS.md`` in place; every hand-written section is kept.
    Returns True when the file changed."""
    path = Path(path)
    doc = path.read_text()
    m2 = re.search(r'^## 2\. .*$', doc, re.M)
    m3 = re.search(r'^## 3\. .*$', doc, re.M)
    m4 = re.search(r'^## 4\. .*$', doc, re.M)
    if not (m2 and m3 and m4 and m2.start() < m3.start() < m4.start()):
        raise ValueError(f"{path}: section headings '## 2.', '## 3.', '## 4.' "
                         "not found in order")
    new = (doc[:m2.end()] + '\n\n' + axes_markdown() + '\n'
           + m3.group(0) + '\n\n' + presets_markdown(PRESETS) + '\n\n'
           + doc[m4.start():])
    if new == doc:
        return False
    path.write_text(new)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ddgclib.methods',
                                description=__doc__)
    p.add_argument('--markdown', action='store_true',
                   help='print the axis tables + preset matrix (METHODS.md source)')
    p.add_argument('--json', action='store_true',
                   help='print every preset as JSON')
    p.add_argument('--preset', metavar='NAME',
                   help='describe one preset (add --markdown for a table)')
    p.add_argument('--update', metavar='METHODS_MD',
                   help='rewrite the generated sections 2-3 of METHODS.md in place')
    a = p.parse_args(argv)

    if a.update:
        changed = update_methods_md(a.update)
        print(f"{a.update}: {'updated' if changed else 'already current'}")
        return 0

    if a.preset:
        m = preset(a.preset)
        print(config_markdown(m) if a.markdown else m.describe())
        return 0
    if a.json:
        print(json.dumps({k: v.to_dict() for k, v in PRESETS.items()}, indent=2))
        return 0
    if a.markdown:
        print(axes_markdown())
        print('## presets')
        print('')
        print(presets_markdown(PRESETS))
        return 0

    print('Method axes (explicit = SolverMethods field, reported = resolved on the mesh):')
    for ax in AXES.values():
        kind = 'explicit' if ax.explicit else 'reported'
        vals = ', '.join(repr(k) for k in ax.keys())
        print(f"  {ax.name:<20} [{kind:<8}] default={ax.default!r:<18} values: {vals}")
    print('\nPresets:')
    for name, m in PRESETS.items():
        print(f"  {name:<40} {m.label}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
