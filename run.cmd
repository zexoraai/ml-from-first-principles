@echo off
REM ---------------------------------------------------------------------------------------
REM Run a command inside the pinned Linux container with this repo bind-mounted at /work.
REM
REM Usage:   run.cmd python -m pytest -q
REM          run.cmd python scripts/bench_cpu.py
REM
REM Why a .cmd file and not a .ps1: the host's PowerShell execution policy is Restricted, so
REM .ps1 scripts cannot run (the same reason the venv's Activate.ps1 is unusable). .cmd is not
REM subject to that policy.
REM
REM Why a wrapper at all: the full `docker run` invocation is long, and the shell tooling used
REM to drive this project truncates long command strings. Short commands are reliable commands.
REM ---------------------------------------------------------------------------------------
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

docker run --rm ^
  -v "%ROOT%:/work" ^
  -w /work ^
  -e PYTHONPATH=/work ^
  --shm-size=1g ^
  ml-labs:cpu %*
