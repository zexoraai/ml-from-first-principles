# Execution environment for all Python/PyTorch work in this repository.
#
# WHY A CONTAINER (this is not a preference, it is a hard requirement here)
# ------------------------------------------------------------------------
# The development host runs Windows with **Smart App Control enabled and enforcing**
# (HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy -> VerifiedAndReputablePolicyState = 1).
# SAC refuses to load binaries it does not consider verified-and-reputable. PyTorch's Windows
# wheels ship unsigned DLLs, so `import torch` fails on the host with:
#
#     OSError: [WinError 4551] An Application Control policy has blocked this file.
#     Error loading ".venv\Lib\site-packages\torch\lib\shm.dll"
#
# numpy imports fine; torch does not. The alternative "fix" -- turning Smart App Control off --
# is a machine-wide security downgrade that Microsoft documents as difficult or impossible to
# reverse without resetting Windows. We do not touch it. SAC governs Windows PE binaries, not
# Linux ELF binaries running inside the WSL2 VM, so a container sidesteps the block without
# weakening the host. See records/DECISIONS.md D-008.
#
# Side benefit that we get for free: this Dockerfile *is* the dependency record required by
# records/EXPERIMENTS.md. Any run can be reproduced from an image digest.

FROM python:3.12-slim-bookworm

# Pinned digest-free base is intentional: the image is rebuilt rarely and `pip freeze` inside
# the image is captured into each run's evidence directory, which is the reproducibility anchor
# that actually matters for numerical results.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Thread count is pinned so timing measurements are comparable between runs. The host has
# 6 physical cores / 12 threads; oversubscribing hurts, and leaving it to chance makes every
# benchmark unreproducible. Override at run time with -e OMP_NUM_THREADS=N when benchmarking
# thread scaling deliberately.
ENV OMP_NUM_THREADS=6 \
    MKL_NUM_THREADS=6

WORKDIR /work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

# Source is bind-mounted at /work by scripts/run.cmd rather than COPYed, so edits on the host
# are visible immediately without a rebuild.
ENV PYTHONPATH=/work

CMD ["python", "-c", "import torch; print(torch.__version__)"]
