function gb_dcn1_solve(id)
% DC N-1 as a MATPOWER user runs it: rundcpf on the base case, then for
% every outage the case with that branch out of service. Keeps angles and
% from-end flows per outage. Prints whether the base solved, the number of
% outages that did not, and the time taken, measured here.
  global GB
  m = GB.models{id};
  n = numel(m.rows);
  tic;
  r0 = rundcpf(m.mpc, GB.mpopt);
  va = zeros(size(m.mpc.bus, 1), n);
  pf = zeros(size(m.mpc.branch, 1), n);
  failed = 0;
  for k = 1:n
    c = m.mpc;
    c.branch(m.rows(k), 11) = 0;   % BR_STATUS
    r = rundcpf(c, GB.mpopt);
    failed = failed + (r.success ~= 1);
    va(:, k) = r.bus(:, 9);
    pf(:, k) = r.branch(:, 14);
  end
  t = toc;
  GB.results{id} = struct('bus', m.mpc.bus(:, 1), 'va', va, 'pf', pf);
  gb_out('%d %d %.9e', r0.success, failed, t);
end
