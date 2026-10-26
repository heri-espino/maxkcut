"""Execute the original convex G81 SDP with periodic GitHub checkpoint publication.

Run from the repository root after installing the package:
    python scripts/run_g81_vm.py --cpu-threads 16 --checkpoint-seconds 900
"""
from maxkcut.g81_vm import main

if __name__ == "__main__":
    raise SystemExit(main())
