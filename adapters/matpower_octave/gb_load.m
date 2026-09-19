function gb_load(path)
% Reads the case into a MATPOWER case struct with a flat start (every bus
% at 1 p.u. and 0 degrees; runpf puts generator buses at their setpoints).
% Prints the model's id and the time taken, measured here.
  global GB
  tic;
  mpc = loadcase(path);
  mpc.bus(:, 8) = 1;   % VM
  mpc.bus(:, 9) = 0;   % VA
  t = toc;
  % Octave keeps a case file's parsed function cached, and a large one (a
  % 4.5 MB .m) slows every later runpf by ~30 ms (MP-Core's function
  % lookups). The case is loaded; drop the function so one case's size
  % never shows in another's timing.
  [~, name] = fileparts(path);
  clear(name);
  GB.models{end + 1} = mpc;
  gb_out('%d %.9e', numel(GB.models), t);
end
