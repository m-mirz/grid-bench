function gb_sweep_solve(id)
% Every scenario, as a MATPOWER user sweeps them: write the scenario into
% the case struct, runpf (flat start: the struct's voltages are never
% updated), keep the voltages. Prints the number of scenarios that did not
% converge and the time taken, measured here.
  global GB
  m = GB.models{id};
  n = size(m.pd, 1);
  tic;
  mpc = m.mpc;
  vm = zeros(size(mpc.bus, 1), n);
  va = vm;
  failed = 0;
  for k = 1:n
    mpc.bus(m.load_rows, 3) = m.pd(k, :)';
    mpc.bus(m.load_rows, 4) = m.qd(k, :)';
    mpc.gen(m.gen_rows, 2) = m.pg(k, :)';
    r = runpf(mpc, GB.mpopt);
    failed = failed + (r.success ~= 1);
    vm(:, k) = r.bus(:, 8);
    va(:, k) = r.bus(:, 9);
  end
  t = toc;
  GB.results{id} = struct('bus', mpc.bus(:, 1), 'vm', vm, 'va', va);
  gb_out('%d %.9e', failed, t);
end
