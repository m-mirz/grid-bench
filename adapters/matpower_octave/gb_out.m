function gb_out(fmt, varargin)
% Prints result lines for adapters/octave_session.py; a matrix argument
% repeats the format per column, one GB_OUT line each.
  printf(['GB_OUT ' fmt '\n'], varargin{:});
end
