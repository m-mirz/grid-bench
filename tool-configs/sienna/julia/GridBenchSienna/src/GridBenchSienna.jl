"""
The Julia half of adapters/sienna_adapter.py (settings and their
justification are documented there). It lives in the image, not in
adapters/, because it is compiled into it: the workload at the bottom runs
load, solve and solution on a 3-bus case at build time, so every fresh
process (the timed one and the memory measurement alike) starts with native
code, as a deployed Julia application would, instead of spending ~95 s in the
JIT on its first case. Changing this file needs `docker/build.sh sienna`.

It depends on PythonCall (unused here) because juliacall loads PythonCall
first: compiled without it, the cached code is invalidated by PythonCall's
method definitions and recompiled on first use (measured: 41 s instead of
2 s for case14 in a fresh process).
"""
module GridBenchSienna

using Logging
using PowerFlows
using PowerSystems
using PrecompileTools: @compile_workload, @setup_workload
using PythonCall: PythonCall

const PF = PowerFlows
const PSY = PowerSystems

struct Model
    data::PF.ACPowerFlowData
    vm0::Vector{Float64}   # flat start: PQ buses at 1 p.u., PV and slack at their setpoints
    va0::Vector{Float64}   # every angle 0
end

function load(path::AbstractString)
    sys = System(path)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; enhanced_flat_start = false, correct_bustypes = true)
    data = with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
        PF.PowerFlowData(pf, sys)
    end
    vm0 = data.bus_magnitude[:, 1]
    vm0[data.bus_type[:, 1] .== (PSY.ACBusTypes.PQ,)] .= 1.0
    return Model(data, vm0, zeros(length(vm0)))
end

"""Restores the flat start (PowerFlows starts from the voltages held in
`data`, i.e. the previous solution), then solves. PowerFlows iterates while
`i < maxIterations`, so `max_iterations + 1` allows exactly `max_iterations`
Newton steps."""
function solve!(m::Model, tol::Float64, max_iterations::Int)
    m.data.bus_magnitude[:, 1] .= m.vm0
    m.data.bus_angles[:, 1] .= m.va0
    return PF.solve_power_flow!(m.data; tol = tol, maxIterations = max_iterations + 1)
end

"""Bus numbers (MATPOWER's, as PowerSystems keeps them), |V| in p.u., angle in degrees."""
function solution(m::Model)
    lookup = PF.get_bus_lookup(m.data)
    numbers = collect(keys(lookup))
    ix = [lookup[n] for n in numbers]
    return numbers, m.data.bus_magnitude[ix, 1], rad2deg.(m.data.bus_angles[ix, 1])
end

"""Versions of the packages that make up the tool, for the results' metadata."""
versions() = Dict(
    "PowerFlows" => pkgversion(PowerFlows),
    "PowerSystems" => pkgversion(PowerSystems),
    "PowerNetworkMatrices" => pkgversion(PF.PNM),
    "InfrastructureSystems" => pkgversion(PSY.IS),
)

# Info and warning logs off: PowerFlows logs at info level on every solve,
# which would be timed. Errors still print.
quiet() = Logging.disable_logging(Logging.Warn)

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
        1 0  0 300 -300 1.02 100 1 250 0 0 0 0 0 0 0 0 0 0 0 0;
        2 50 0 300 -300 1.01 100 1 250 0 0 0 0 0 0 0 0 0 0 0 0;
    ];
    mpc.branch = [
        1 2 0.01 0.1 0.02 0 0 0 0    0 1 -360 360;
        2 3 0.01 0.1 0.02 0 0 0 0.98 0 1 -360 360;
        1 3 0.02 0.2 0.04 0 0 0 0    0 1 -360 360;
    ];
    """
    path = joinpath(mktempdir(), "gb3.m")
    write(path, case)
    @compile_workload begin
        Logging.with_logger(Logging.NullLogger()) do
            m = load(path)
            solve!(m, 1e-8, 30)
            solution(m)
        end
    end
end

end
