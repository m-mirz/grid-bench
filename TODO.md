# TODO

## Faster runs

Measured on the 2026-10-31 GPU run (A100 machine, 30 cores): the timed
rounds are 3 to 50 % of a run's wall time.

- The memory measurement (`adapters/memory.py`) spawns a fresh process per
  case. For a Julia tool that is Julia's start and package load, for a GPU
  tool also the first solve's kernel compilation: examodelspower_gpu 9 s
  import + 89 s first solve (case14), so about 30 of its 47 minutes for 18
  cases; exapf_gpu's power flow 751 s wall for 27 s timed. Profile the
  other tools likewise, then consider one memory child per session that
  measures every case in turn (load, solve, release, reset `VmHWM`),
  after checking that a case's peak does not carry over to the next.
- The first solve test of a session can get one round: its warm-up
  includes the compilation, exceeds `SLOW_SOLVE_SECONDS` and skips the
  calibration (examodelspower_gpu's case14 OPF: 1 round, 385 ms, against
  81 ms the run before). A second, untimed warm-up when the first exceeds
  the threshold would separate compilation from a slow solve.
- `SLOW_SOLVE_SECONDS = 30`: a 29 s solve still gets a warm-up and 5
  rounds; 57 passing tests take over 60 s in total. 10 s would cut those
  with no visible change in a median.
- `docker/run_single.sh` runs every problem a tool has. A CPU baseline in
  `results-docker/gpu/` only needs power flow, batch and N-1; the rest
  (lightsim2grid's DC, gridoxide's SE, OPF on the large PGLib cases with
  Ipopt, DC) is written there and deleted. A `--problems` option.
- A CPU-optimized machine for the CPU tools: rerun all of them there (the
  report stops on a mix), with the large cases for the fastest, since most
  have not run them. Then publish a baseline's ratio between the two
  machines as a note, not to rescale anything.
- gridoxide_gpu's results report version 0.3.0: built from 1dd1c96, which
  differs from v0.4.0 only in its version numbers. The next rerun at the
  pinned release corrects the label, and records the commit on DC too.
- `docker/build.sh <tool>` does not build the harness image, which
  `run_single.sh`'s prep needs; without it every input is missing and the
  run records only failures.

## OPF at the limit of 500

- veragrid's pglib_opf_case2869_pegase still carries its failure at the old
  limit of 200 (its rerun hit a 300 s timeout, not the limit). Rerun it at
  pytest's default timeout, up to 30 minutes, and merge it as the other
  limit reruns were (`grid_bench.reruns`).

## GPU tools

- examodelspower_gpu: case13659pegase was accepted in one run and stalled
  in the next with the same pins (iteration counts on the device vary by 1
  or 2 between runs). Find where the run-to-run difference comes from
  (cuDSS?) before reading much into one run's verdict on it.
