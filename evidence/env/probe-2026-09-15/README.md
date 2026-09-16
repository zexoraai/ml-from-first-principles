# Environment probe — 2026-09-15

Raw transcripts from the initial environment inspection, preserved because two of them are the
primary evidence behind decisions D-004 (no CUDA → Triton executed externally) and D-008 (Smart
App Control → containerised execution).

| File | What it establishes |
|---|---|
| `hardware.txt` | CPU model/core count, RAM, GPU, free disk |
| `toolchain.txt` | Python/pip/git/node/npm versions, preinstalled packages, `gh` auth state |
| `no-cuda.txt` | `nvidia-smi` absent; GPU is integrated AMD Radeon |
| `smart-app-control-block.txt` | **The `import torch` failure and its root cause.** Basis for D-008 |
| `wsl-docker.txt` | WSL2 + Docker Desktop present and working — the escape route |

Commands were run in Windows PowerShell on the host. Output encoding is UTF-16LE in places
because PowerShell redirection defaults to it; content is unmodified otherwise.
