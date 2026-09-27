function gb_opf_solution(id)
% Bus lines "B number |V| angle", then generator lines "G row PG QG" for
% every online generator (row 0-based, as in the .m), of the last solve.
  global GB
  r = GB.results{id};
  gb_out('B %d %.17g %.17g', [r.bus(:, 1) r.bus(:, 8) r.bus(:, 9)]');
  on = find(r.gen(:, 8) > 0);
  gb_out('G %d %.17g %.17g', [on - 1, r.gen(on, 2), r.gen(on, 3)]');
end
