"""Launch SGLang after installing the RWKV-7 external registrations."""

from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ.setdefault(
        "SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_rwkv7_ascend.models"
    )
    from sglang_rwkv7_ascend import register

    register()
    # RWKV's recurrent state is carried by the MambaPool.  The custom no-op
    # backend avoids constructing an empty full-attention KV pool.
    if not any(a == "--attention-backend" or a.startswith("--attention-backend=") for a in sys.argv[1:]):
        sys.argv.extend(["--attention-backend", "rwkv7_ascend"])
    from sglang.launch_server import main as launch_main

    launch_main()


if __name__ == "__main__":
    main()
