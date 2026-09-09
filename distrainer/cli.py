"""distrainer command line: inspect / resume / export checkpoints (spec section 6.4)."""

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print("distrainer CLI is not implemented yet; see docs/distrainer-spec.md section 6.4.")
    return 0 if not argv else 2


if __name__ == "__main__":
    raise SystemExit(main())
