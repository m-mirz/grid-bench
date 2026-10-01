function gb_sweep_load(path, sweep_file)
% A batch case: the case struct with a flat start, as gb_load, and its
% scenarios from a .mat the Python side wrote (adapters/matpower_batch_adapter.py):
% load_rows, gen_rows (1-based rows of mpc.bus and mpc.gen), pd, qd, pg
% (scenarios x those rows). Prints the model's id and the time taken.
  global GB
  tic;
  mpc = loadcase(path);
  mpc.bus(:, 8) = 1;   % VM
  mpc.bus(:, 9) = 0;   % VA
  s = load(sweep_file);
  t = toc;
  [~, name] = fileparts(path);
  clear(name);
  GB.models{end + 1} = struct('mpc', mpc, 'load_rows', s.load_rows(:), 'gen_rows', s.gen_rows(:), ...
                              'pd', s.pd, 'qd', s.qd, 'pg', s.pg);
  gb_out('%d %.9e', numel(GB.models), t);
end
