function gb_solve(id)
% One power flow, as MATPOWER runs it: runpf on the case struct, which is
% left unchanged, so every solve starts flat. Prints success, iterations
% and the time taken, measured here.
  global GB
  tic;
  r = runpf(GB.models{id}, GB.mpopt);
  t = toc;
  GB.results{id} = r;
  gb_out('%d %d %.9e', r.success, r.iterations, t);
end
