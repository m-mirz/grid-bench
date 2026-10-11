function gb_dc_solve(id)
% One DC power flow, as MATPOWER runs it: rundcpf on the case struct.
% Keeps bus angles (degrees) and from-end branch flows (MW, PF column).
% Prints success and the time taken, measured here.
  global GB
  tic;
  r = rundcpf(GB.models{id}, GB.mpopt);
  t = toc;
  GB.results{id} = struct('bus', r.bus(:, 1), 'va', r.bus(:, 9), 'pf', r.branch(:, 14));
  gb_out('%d %.9e', r.success, t);
end
