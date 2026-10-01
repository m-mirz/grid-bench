function gb_n1_solve(id)
% N-1 as a MATPOWER user runs it: runpf on the base case (flat start),
% then for every outage the case with that branch out of service, started
% from the base solution (runpf starts from the struct's VM and VA).
% Prints whether the base converged, the number of outages that did not,
% and the time taken, measured here.
  global GB
  m = GB.models{id};
  n = numel(m.rows);
  tic;
  r0 = runpf(m.mpc, GB.mpopt);
  start = m.mpc;
  start.bus(:, 8) = r0.bus(:, 8);
  start.bus(:, 9) = r0.bus(:, 9);
  vm = zeros(size(start.bus, 1), n);
  va = vm;
  failed = 0;
  for k = 1:n
    c = start;
    c.branch(m.rows(k), 11) = 0;   % BR_STATUS
    r = runpf(c, GB.mpopt);
    failed = failed + (r.success ~= 1);
    vm(:, k) = r.bus(:, 8);
    va(:, k) = r.bus(:, 9);
  end
  t = toc;
  GB.results{id} = struct('bus', start.bus(:, 1), 'vm', vm, 'va', va);
  gb_out('%d %d %.9e', r0.success, failed, t);
end
