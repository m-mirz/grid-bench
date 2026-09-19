function gb_free(id)
% Drops a model the Python side no longer holds.
  global GB
  GB.models{id} = [];
  if numel(GB.results) >= id
    GB.results{id} = [];
  end
end
