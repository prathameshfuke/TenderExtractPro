"""
Deprecated helper retained only to prevent accidental overwrites.

Historically this script regenerated extraction.py from an older prompt/schema
template. The runtime pipeline has since moved on, so re-running the old
generator would silently corrupt the current extraction schema.
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "This generator has been retired because it no longer matches the live "
        "extraction schema. Update tender_extraction/extraction.py directly.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
