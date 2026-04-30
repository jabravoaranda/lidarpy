# AGENTS

## Purpose

This repository is being migrated from the lidar submodule of `gfatpy` into a
standalone package named `lidarpy`.

This file is for Codex coordination only. It must be tracked by git, but it
must not be shipped in PyPI distributions.

## Required Workflow

1. Read `ROADMAP.md` before changing code.
2. Work on one roadmap task at a time.
3. Update `ROADMAP.md` when a task changes state, a blocker is found, or a
   decision changes the migration plan.
4. Prefer small commits with clear messages.
5. Do not reintroduce removed legacy modules unless `ROADMAP.md` explicitly
   says so.
6. Keep tests compatible with Windows and Linux.
7. Before long test runs, clean local pytest/unzip temporaries and check disk
   space.

## Local Test Pattern

Use local temp directories so Windows does not fail on permission-heavy system
temp paths:

```powershell
Remove-Item .pytest_tmp,.pytest_cache,tmp_unzipped_* -Recurse -Force -ErrorAction SilentlyContinue; $env:PYTHONPATH='src'; $env:MPLBACKEND='Agg'; .\.venv311\Scripts\python -m pytest <tests> -q
```

Run large test groups in chunks. The RS/DC fixtures generate large NetCDF files
and can fill small disks quickly.

## Packaging Note

`AGENTS.md` and `ROADMAP.md` are coordination files. Keep them in git, but do
not add them to `tool.hatch.build.targets.sdist.include` or package data.
