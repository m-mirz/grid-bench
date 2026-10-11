function gb_dc_solution(id, path)
% The last DC solve's bus numbers, angles (degrees) and from-end flows (MW,
% one row per branch row), one column per outage for DC N-1, to a .mat.
  global GB
  r = GB.results{id};
  bus = r.bus; va = r.va; pf = r.pf;
  save('-v7', path, 'bus', 'va', 'pf');
end
