function gb_opf_solve(id)
% One AC-OPF, as MATPOWER runs it: runopf on the case struct. Prints
% success, iterations and the time taken, measured here.
  global GB
  tic;
  r = runopf(GB.models{id}, GB.opfopt);
  t = toc;
  GB.results{id} = r;
  gb_out('%d %d %.9e', r.success, r.raw.output.iterations, t);
end
