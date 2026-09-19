function gb_init(matpower_dir, tol, max_it)
% Puts MATPOWER on the path for this session (not saved) and fixes the
% power-flow options every solve uses (see adapters/matpower_adapter.py).
  global GB
  here = pwd; cd(matpower_dir); install_matpower(1, 0, 0); cd(here);
  GB.mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.alg', 'NR', 'pf.tol', tol, ...
                      'pf.nr.max_it', max_it, 'pf.enforce_q_lims', 0);
  % Octave parses a function file on its first call: run runpf once on
  % MATPOWER's own case9, so that parsing is part of the memory baseline (as
  % importing a Python tool's modules is) rather than charged to a case.
  runpf(loadcase('case9'), GB.mpopt);
  GB.models = {};
  GB.results = {};
  gb_out('%s %s', mpver, version);
end
