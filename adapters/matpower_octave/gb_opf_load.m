function gb_opf_load(path)
% Reads the case with the common flat start written into it (every bus at
% 1 p.u. and 0 degrees, every generator at the middle of its P and Q
% ranges); runopf starts from it (opf.start = 2) and leaves it unchanged,
% so every solve starts there. Prints the model's id and the time taken.
  global GB
  tic;
  mpc = loadcase(path);
  mpc.bus(:, 8) = 1;                                   % VM
  mpc.bus(:, 9) = 0;                                   % VA
  mpc.gen(:, 2) = (mpc.gen(:, 9) + mpc.gen(:, 10)) / 2;   % PG = (PMAX + PMIN) / 2
  mpc.gen(:, 3) = (mpc.gen(:, 4) + mpc.gen(:, 5)) / 2;    % QG = (QMAX + QMIN) / 2
  t = toc;
  [~, name] = fileparts(path);
  clear(name);   % see gb_load
  GB.models{end + 1} = mpc;
  gb_out('%d %.9e', numel(GB.models), t);
end
