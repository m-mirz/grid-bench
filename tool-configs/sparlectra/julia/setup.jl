# The Julia counterpart of `uv lock` / `uv sync --frozen` for the sparlectra image.
#
#     julia setup.jl lock    <project dir>   # resolve Project.toml into Manifest.toml
#     julia setup.jl install <project dir>   # install exactly Manifest.toml, precompile
#
# Both resolve against one snapshot of the General registry. Unlike every
# other pin (sienna's snapshot, `exclude-newer = "P7D"`), this one is NOT 7
# days old: it is the commit that registered Sparlectra 0.17.3, taken on
# 2026-09-25 by deliberate choice, to benchmark the newest release of a
# package that releases almost daily. Everything it resolves may therefore be
# up to that fresh. Packages are still fetched by the content hash
# (git-tree-sha1) recorded in Manifest.toml and verified on download.
using Pkg

const REGISTRY_COMMIT = "f585b0863023d863247a924ae3f7aeda2f31713c"   # General, 2026-09-24T18:39Z: registers Sparlectra 0.17.3

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
