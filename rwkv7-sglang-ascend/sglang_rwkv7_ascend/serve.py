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
        # Reuse SGLang's accepted NPU backend name.  Plugin registration maps
        # it to the all-linear no-op full-attention half in this process.
        sys.argv.extend(["--attention-backend", "ascend"])
    # The lightweight state backend intentionally supports eager scheduling;
    # graph capture is disabled until its fixed-address replay path is verified.
    if "--disable-cuda-graph" not in sys.argv[1:]:
        sys.argv.append("--disable-cuda-graph")
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args

    run_server(prepare_server_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
