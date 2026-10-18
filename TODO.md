# TODO

## Next GPU run

Rerun on the GPU machine at a clean commit, into `results-docker/gpu/`:

```bash
GRID_BENCH_RESULTS=gpu docker/run_single.sh exapf_gpu --groups smoke,scaling,feature,large
GRID_BENCH_RESULTS=gpu docker/run_single.sh gpusim2grid --groups smoke,scaling,feature,large
GRID_BENCH_RESULTS=gpu docker/run_single.sh examodelspower_gpu --groups smoke,scaling,feature,large
docker compose -f docker/docker-compose.yml run --rm reports
```

- exapf_gpu: the published batch and N-1 import times include the memory
  check (`available_bytes`: GC and `CUDA.reclaim()`, case14#sweep 825 ms
  instead of 38), now run outside the timed rounds.
- gpusim2grid (`-batch.json`, `-n1.json`) recorded `eb43a78-dirty`, and
  examodelspower_gpu (`-opf.json`) recorded `e422d35-dirty`. The content is
  right (`batch_size 2048`, `bound_relax_factor 1e-8`); this is for provenance.
- Afterwards, check the docstring figures against the new JSON:
  `adapters/gpusim2grid_batch_adapter.py`, `gpusim2grid_n1_adapter.py`,
  `exapf_gpu_batch_adapter.py`, `exapf_gpu_n1_adapter.py`,
  `examodelspower_gpu_opf_adapter.py`.
