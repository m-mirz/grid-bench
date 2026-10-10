function gb_sweep_solve(id)
% Every scenario, as a MATPOWER user sweeps them from a solved base case:
% runpf on the base case (flat start), then every scenario written into the
% case struct and started from the base solution (runpf starts from the
% struct's VM and VA), keeping the voltages. Prints whether the base
% converged, the number of scenarios that did not, and the time taken,
% measured here.
  global GB
  m = GB.models{id};
  n = size(m.pd, 1);
  tic;
  r0 = runpf(m.mpc, GB.mpopt);
  mpc = m.mpc;
  mpc.bus(:, 8) = r0.bus(:, 8);
  mpc.bus(:, 9) = r0.bus(:, 9);
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
  gb_out('%d %d %.9e', r0.success, failed, t);
end
