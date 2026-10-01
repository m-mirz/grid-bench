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

"""
A batch: one `PowerFlowData` with a time step per scenario, the form in which
PowerFlows solves several operating points of one system
(`solve_power_flow!` over its time steps, one after another). Every column is
seeded from the system; each scenario's change of injections and withdrawals
per bus (p.u. on the system base, rows `numbers`, MATPOWER bus numbers) is
added to its column. `vm0`, `va0`: the flat start of every column.
"""
struct Batch
    data::PF.ACPowerFlowData
    vm0::Matrix{Float64}
    va0::Matrix{Float64}
end

function load_batch(path::AbstractString, numbers::AbstractVector, dp_injection::AbstractMatrix,
                    dp_withdrawal::AbstractMatrix, dq_withdrawal::AbstractMatrix)
    sys = System(path)
    n = size(dp_injection, 2)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = false, correct_bustypes = true, time_steps = n)
    data = with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
        PF.PowerFlowData(pf, sys)
    end
    lookup = PF.get_bus_lookup(data)
    ix = [lookup[Int(b)] for b in numbers]
    data.bus_active_power_injections[ix, :] .+= dp_injection
    data.bus_active_power_withdrawals[ix, :] .+= dp_withdrawal
    data.bus_reactive_power_withdrawals[ix, :] .+= dq_withdrawal
    vm0 = copy(data.bus_magnitude)
    vm0[data.bus_type .== (PSY.ACBusTypes.PQ,)] .= 1.0
    return Batch(data, vm0, zeros(size(vm0)))
end

"""Restores the flat start of every time step, solves them all; true if every one converged."""
function solve_batch!(b::Batch, tol::Float64, max_iterations::Int)
    b.data.bus_magnitude .= b.vm0
    b.data.bus_angles .= b.va0
    PF.solve_power_flow!(b.data; tol = tol, maxIterations = max_iterations + 1)
    return count(!, b.data.converged)
end

"""Bus numbers, |V| in p.u. and angle in degrees, buses x time steps."""
function batch_solution(b::Batch)
    lookup = PF.get_bus_lookup(b.data)
    numbers = collect(keys(lookup))
    ix = [lookup[n] for n in numbers]
    return numbers, b.data.bus_magnitude[ix, :], rad2deg.(b.data.bus_angles[ix, :])
end

"""
N-1 on one system, for the contingency benchmark
(adapters/sienna_n1_adapter.py). PowerFlows takes a branch out only through
the system: the branch made unavailable and a new `PowerFlowData` built
(PowerNetworkMatrices can modify a Ybus for an outage, but PowerFlows' AC
solve never takes a modified one). The base case is solved from the common
flat start; every outage's data starts from the base solution, by bus
number; the branch is made available again afterwards. `rows`: 0-based
branch rows; PowerSystems names each branch `...-i_<row + 1>`, asserted by
its buses.
"""
struct N1
    sys::System
    pf::ACPowerFlow{NewtonRaphsonACPowerFlow}
    branches::Vector{PSY.ACBranch}
    vm::Vector{Matrix{Float64}}   # [numbers; vm per outage], written by solve_n1!
end

function load_n1(path::AbstractString, rows::AbstractVector, from::AbstractVector, to::AbstractVector)
    sys = System(path)
    row_of(b) = parse(Int, last(split(PSY.get_name(b), "-i_"))) - 1
    by_row = Dict(row_of(b) => b for b in PSY.get_components(PSY.ACBranch, sys))
    branches = [by_row[Int(r)] for r in rows]
    for (b, f, t) in zip(branches, from, to)
        arc = PSY.get_arc(b)
        @assert (PSY.get_number(PSY.get_from(arc)), PSY.get_number(PSY.get_to(arc))) == (Int(f), Int(t))
    end
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; enhanced_flat_start = false, correct_bustypes = true)
    return N1(sys, pf, branches, Matrix{Float64}[])
end

function _data(n::N1)
    return with_units_base(n.sys, PSY.UnitSystem.SYSTEM_BASE) do
        PF.PowerFlowData(n.pf, n.sys)
    end
end

"""The base case and every outage; returns the number that did not
converge (-1: the base case did not)."""
function solve_n1!(n::N1, tol::Float64, max_iterations::Int)
    data = _data(n)
    vm0 = data.bus_magnitude[:, 1]
    vm0[data.bus_type[:, 1] .== (PSY.ACBusTypes.PQ,)] .= 1.0
    base = Model(data, vm0, zeros(length(vm0)))
    solve!(base, tol, max_iterations) || return -1
    numbers, vm_base, va_base = solution(base)
    start = Dict(b => (m, deg2rad(a)) for (b, m, a) in zip(numbers, vm_base, va_base))
    vms, vas, failed = Matrix{Float64}(undef, length(numbers), length(n.branches)), similar(vm_base, length(numbers), length(n.branches)), 0
    for (k, b) in enumerate(n.branches)
        PSY.set_available!(b, false)
        d = _data(n)
        lookup = PF.get_bus_lookup(d)
        for (bus, i) in lookup
            d.bus_magnitude[i, 1], d.bus_angles[i, 1] = start[bus]
        end
        failed += !PF.solve_power_flow!(d; tol = tol, maxIterations = max_iterations + 1)
        PSY.set_available!(b, true)
        ix = [lookup[bus] for bus in numbers]
        vms[:, k] = d.bus_magnitude[ix, 1]
        vas[:, k] = rad2deg.(d.bus_angles[ix, 1])
    end
    empty!(n.vm)
    push!(n.vm, reshape(Float64.(numbers), :, 1), vms, vas)
    return failed
end

"""Bus numbers, |V| and angle in degrees, buses x outages, of the last solve_n1!."""
n1_solution(n::N1) = (Int.(n.vm[1][:, 1]), n.vm[2], n.vm[3])

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
            d = [0.0 0.1; 0.0 -0.05]
            b = load_batch(path, [2, 3], d, -d, 0.5 .* d)
            solve_batch!(b, 1e-8, 30)
            batch_solution(b)
            o = load_n1(path, [0, 2], [1, 1], [2, 3])
            solve_n1!(o, 1e-8, 30)
            n1_solution(o)
        end
    end
end

end
