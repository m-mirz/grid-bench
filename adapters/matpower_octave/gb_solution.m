function gb_solution(id)
% Bus number, |V| in p.u. and angle in degrees of the last solve.
  global GB
  b = GB.results{id}.bus;
  gb_out('%d %.17g %.17g', [b(:, 1) b(:, 8) b(:, 9)]');
end
