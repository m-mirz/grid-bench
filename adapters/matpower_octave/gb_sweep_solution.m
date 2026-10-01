function gb_sweep_solution(id, out_file)
% Bus numbers, |V| in p.u. and angle in degrees (buses x scenarios) of the
% last sweep, saved to a .mat for the Python side: a million numbers do not
% belong on the text bridge.
  global GB
  r = GB.results{id};
  bus = r.bus; vm = r.vm; va = r.va;
  save('-v7', out_file, 'bus', 'vm', 'va');
  gb_out('%d', numel(bus));
end
