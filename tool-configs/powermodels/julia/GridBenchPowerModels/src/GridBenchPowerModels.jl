"""
The Julia half of adapters/powermodels_opf_adapter.py (settings and their
justification are documented there). It lives in the image, not in
adapters/, because it is compiled into it: the workload at the bottom runs
load, solve and solution on a 3-bus case at build time, so every fresh
process starts with native code instead of spending its first case in the
JIT. Changing this file needs `docker/build.sh powermodels`.

It depends on PythonCall (unused here) for the same reason as
GridBenchSienna: juliacall loads PythonCall first, and code compiled without
it is invalidated by PythonCall's method definitions.
"""
module GridBenchPowerModels

using Logging
using PrecompileTools: @compile_workload, @setup_workload
using PythonCall: PythonCall
import Ipopt
import JuMP
import PowerModels as PM

struct Model
    pm::PM.AbstractPowerModel
    base_mva::Float64
end

"""Parses the `.m` (PowerModels' own MATPOWER parser), writes the common
flat start into the data (every bus at 1 p.u. and 0 rad, every generator at
the middle of its P and Q ranges), and builds the AC-OPF once: a JuMP model
of the polar formulation, with Ipopt attached. `solve!` re-optimizes it."""
function load(path::AbstractString, tol::Float64, max_iterations::Int)
    data = PM.parse_file(String(path))
    for bus in values(data["bus"])
        bus["vm_start"] = 1.0
        bus["va_start"] = 0.0
    end
    for gen in values(data["gen"])
        gen["pg_start"] = (gen["pmin"] + gen["pmax"]) / 2
        gen["qg_start"] = (gen["qmin"] + gen["qmax"]) / 2
    end
    pm = PM.instantiate_model(data, PM.ACPPowerModel, PM.build_opf)
    JuMP.set_optimizer(pm.model, JuMP.optimizer_with_attributes(Ipopt.Optimizer,
        "tol" => tol, "max_iter" => max_iterations, "linear_solver" => "mumps", "print_level" => 0))
    return Model(pm, data["baseMVA"])
end

"""One solve from the start values set in `load` (JuMP passes them to Ipopt
on every `optimize!`, so every solve starts flat). Returns Ipopt's
iteration count, or -1 if not locally solved."""
function solve!(m::Model)
    JuMP.optimize!(m.pm.model)
    JuMP.termination_status(m.pm.model) == JuMP.LOCALLY_SOLVED || return -1
    return Int(JuMP.MOI.get(m.pm.model, JuMP.MOI.BarrierIterations()))
end

"""Bus numbers, |V| in p.u., angles in degrees; then generator rows (0-based,
the `.m`'s order), their buses, P in MW and Q in MVAr, of the last solve."""
function solution(m::Model)
    ref = PM.ref(m.pm)
    buses = sort(collect(keys(ref[:bus])))
    gens = sort(collect(keys(ref[:gen])))
    value(name, i) = JuMP.value(PM.var(m.pm, name, i))   # the model's variables, per unit and radians
    return (string.(buses),
            [value(:vm, i) for i in buses],
            [rad2deg(value(:va, i)) for i in buses],
            [ref[:gen][i]["index"] - 1 for i in gens],
            [ref[:gen][i]["gen_bus"] for i in gens],
            [value(:pg, i) * m.base_mva for i in gens],
            [value(:qg, i) * m.base_mva for i in gens])
end

"""Versions of the packages that make up the tool, for the results' metadata."""
versions() = Dict(
    "PowerModels" => pkgversion(PM),
    "JuMP" => pkgversion(JuMP),
    "Ipopt.jl" => pkgversion(Ipopt),
    "Ipopt_jll" => pkgversion(Ipopt.Ipopt_jll),   # 300.1400.1902 is Ipopt 3.14.19
)

# PowerModels logs every data correction through Memento; the parse is timed.
quiet() = (PM.silence(); Logging.disable_logging(Logging.Warn))

@setup_workload begin
    case = """
    function mpc = gb3
    mpc.version = '2';
    mpc.baseMVA = 100;
    mpc.bus = [
        1 3 0  0  0 0 1 1 0 110 1 1.1 0.9;
        2 2 0  0  0 0 1 1 0 110 1 1.1 0.9;
        3 1 90 30 0 0 1 1 0 110 1 1.1 0.9;
    ];
    mpc.gen = [
        1 0  0 300 -300 1.02 100 1 250 0;
        2 50 0 300 -300 1.01 100 1 250 0;
    ];
    mpc.gencost = [
        2 0 0 3 0.01 10 0;
        2 0 0 3 0.02 12 0;
    ];
    mpc.branch = [
        1 2 0.01 0.1 0.02 250 250 250 0    0 1 -30 30;
        2 3 0.01 0.1 0.02 250 250 250 0.98 0 1 -30 30;
        1 3 0.02 0.2 0.04 250 250 250 0    0 1 -30 30;
    ];
    """
    path = joinpath(mktempdir(), "gb3.m")
    write(path, case)
    @compile_workload begin
        PM.silence()
        Logging.with_logger(Logging.NullLogger()) do
            redirect_stdout(devnull) do
                m = load(path, 1e-6, 200)
                solve!(m)
                solution(m)
            end
        end
    end
end

end
