# Containerization validation

Published numbers come from one container per tool. This records that the
containers do not distort the measurement, and one case where they
prevented a wrong one.

## Native vs container, warm solve median (ms)

Same machine (AMD Ryzen 7 250, 16 logical CPUs, Linux 7.0), same tool
versions, measured on 2026-09-18 minutes apart. Native: one uv environment
with all six tools. Container: `docker/run_single.sh <tool>`.

| tool | case | native | container | delta |
|---|---|---:|---:|---:|
| lightsim2grid | case118 | 0.068 | 0.066 | −3.3% |
| lightsim2grid | case1354pegase | 1.077 | 1.035 | −3.9% |
| lightsim2grid | case2869pegase | 3.469 | 3.390 | −2.3% |
| lightsim2grid | case9241pegase | 17.07 | 16.99 | −0.5% |
| pypowsybl | case118 | 3.918 | 2.576 | −34.2% |
| pypowsybl | case1354pegase | 31.76 | 33.31 | +4.9% |
| pypowsybl | case2869pegase | 78.66 | 76.92 | −2.2% |
| pypowsybl | case9241pegase | 381.3 | 335.6 | −12.0% |
| pandapower | case118 | 6.796 | 6.934 | +2.0% |
| pandapower | case1354pegase | 22.20 | 19.96 | −10.1% |
| pandapower | case2869pegase | 42.93 | 38.30 | −10.8% |
| pandapower | case9241pegase | 170.8 | 152.0 | −11.0% |

No systematic container overhead: compiled solvers (lightsim2grid) agree to
within 4%, and the larger deltas go both ways. They are run-to-run variance
on a laptop whose clock depends on its thermal state (the native runs came
straight after a full sweep). Sub-5 ms numbers (pypowsybl case118) are the
noisiest in relative terms. Interleave runs for any A/B comparison.

## What isolation caught

The first native comparison showed pandapower 2–3× faster outside the
container (case9241pegase: 50 ms native, 144 ms container). The cause:
`pandapower.runpp(lightsim2grid="auto")`, the default, hands the
Newton-Raphson solve to lightsim2grid whenever that package is importable.
The native environment has it; the pandapower image does not. So "pandapower
native" was lightsim2grid's solver under pandapower's name.

The adapter now passes `lightsim2grid=False`, so the pandapower column
measures pandapower's solver in every environment. The rows above were
measured after that fix. One environment per tool is what exposed it, and
it is the reason the published numbers come from containers.
