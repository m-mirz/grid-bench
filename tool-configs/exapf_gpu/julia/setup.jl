# The Julia counterpart of `uv lock` / `uv sync --frozen` for the exapf_gpu image.
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

"""The CUDA libraries are lazy artifacts: downloaded when a package first
loads them, which precompilation does, for the runtime version
LocalPreferences.toml fixes. A container has no network, so the build fails
here if any of them is not installed."""
function check_cuda_artifacts()
    for (name, uuid) in ("CUDA_Runtime_jll" => "76a88914-d11a-5bdc-97e0-2f5a05c973a2",
                         "CUDSS_jll" => "4889d778-9329-5762-9fec-0578a5d30366",
                         "CUDA_Compiler_jll" => "d1e2174e-dfdc-576e-b43e-73b79eb1aca8")
        jll = Base.require(Base.PkgId(Base.UUID(uuid), name))
        available = Base.invokelatest(jll.is_available)   # loaded after this function was compiled
        available && isdir(jll.artifact_dir) || error("$name has no artifact for $(jll.host_platform)")
        println("$name: ", jll.artifact_dir)
    end
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
    check_cuda_artifacts()
else
    error("mode must be lock or install, got $mode")
end
