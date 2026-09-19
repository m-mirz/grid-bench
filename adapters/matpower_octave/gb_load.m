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
  GB.models{end + 1} = mpc;
  gb_out('%d %.9e', numel(GB.models), t);
end
