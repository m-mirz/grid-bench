# The Julia counterpart of `uv lock` / `uv sync --frozen` for the sienna image.
#
#     julia setup.jl lock    <project dir>   # resolve Project.toml into Manifest.toml
#     julia setup.jl install <project dir>   # install exactly Manifest.toml, precompile
#
# Both resolve against one snapshot of the General registry, taken at least 7
# days before it was chosen: the analogue of `exclude-newer = "P7D"` for the
# Python images, so every transitive Julia package is at least that old.
# Packages are then fetched by the content hash (git-tree-sha1) recorded in
# Manifest.toml and verified on download.
using Pkg

const REGISTRY_COMMIT = "8d3c2849dca2836af30a3de65fe3ae34029dfe77"   # General, 2026-09-11T23:48Z

function install_registry_snapshot()
    target = joinpath(first(DEPOT_PATH), "registries", "General")
    isdir(target) && return
    url = "https://github.com/JuliaRegistries/General/archive/$REGISTRY_COMMIT.tar.gz"
    tmp = mktempdir()
    Pkg.PlatformEngines.download_verify_unpack(url, nothing, tmp; ignore_existence = true)
    mkpath(dirname(target))
    mv(joinpath(tmp, "General-$REGISTRY_COMMIT"), target)
end

mode, project = ARGS
install_registry_snapshot()
Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true      # never move past the snapshot
Pkg.activate(project)
if mode == "lock"
    Pkg.resolve()
elseif mode == "install"
    Pkg.instantiate()
    Pkg.precompile()
else
    error("mode must be lock or install, got $mode")
end
