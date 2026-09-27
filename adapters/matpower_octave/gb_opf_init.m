function gb_opf_init(tol, max_it)
% Fixes the AC-OPF options every solve uses (see adapters/matpower_opf_adapter.py).
% Called once, after gb_init has put MATPOWER on the path.
  global GB
  GB.opfopt = mpoption('verbose', 0, 'out.all', 0, 'model', 'AC', 'opf.ac.solver', 'MIPS', ...
                       'opf.start', 2, 'opf.flow_lim', 'S', 'opf.ignore_angle_lim', 0, ...
                       'mips.feastol', tol, 'mips.gradtol', tol, 'mips.comptol', tol, ...
                       'mips.costtol', tol, 'mips.max_it', max_it, 'mips.step_control', 0);
  % As gb_init does for runpf: parse runopf's function files now, so that
  % is part of the memory baseline rather than charged to a case.
  runopf(loadcase('case9'), GB.opfopt);
end
