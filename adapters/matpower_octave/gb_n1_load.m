function gb_n1_load(path, rows)
% A contingency case: the case struct with a flat start, as gb_load, and
% the branch rows of its outages (1-based). Prints the model's id and the
% time taken.
  global GB
  tic;
  mpc = loadcase(path);
  mpc.bus(:, 8) = 1;   % VM
  mpc.bus(:, 9) = 0;   % VA
  t = toc;
  [~, name] = fileparts(path);
  clear(name);
  GB.models{end + 1} = struct('mpc', mpc, 'rows', rows(:));
  gb_out('%d %.9e', numel(GB.models), t);
end
