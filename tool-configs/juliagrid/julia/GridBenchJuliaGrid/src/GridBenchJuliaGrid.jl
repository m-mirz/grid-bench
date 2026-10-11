"""
The Julia half of the JuliaGrid adapters (adapters/juliagrid_adapter.py and
its state-estimation, OPF, batch and N-1 siblings; settings and their
justification are documented there). It lives in the image, not in
adapters/, because it is compiled into it: the workload at the bottom runs
every entry point on a 3-bus MATPOWER case at build time, so every fresh
process starts with native code instead of spending its first case in the
JIT. Changing this file needs `docker/build.sh juliagrid`.

JuliaGrid is imported qualified (`JG.`): it exports `solve!`, which this
module defines for its own types.

It depends on PythonCall (unused here) for the same reason as
GridBenchSienna: juliacall loads PythonCall first, and code compiled without
it is invalidated by PythonCall's method definitions.

Units: JuliaGrid's defaults, per unit on the case's base power and radians;
the Python side converts.
"""
module GridBenchJuliaGrid

import Ipopt
import JuMP
import JuliaGrid as JG
using Logging
using PrecompileTools: @compile_workload, @setup_workload
using PythonCall: PythonCall

"""The case as JuliaGrid reads it, with every bus voltage set to 1 p.u. and
0 rad: the `.m`'s VM and VA columns (often a solved operating point) are
JuliaGrid's starting point otherwise. Generator setpoints are untouched.
`optimal`: also read the OPF data (limits, `gencost`), for the OPF only."""
function flat_system(path::AbstractString; optimal::Bool = false)
    system = JG.powerSystem(String(path); optimal)
    system.bus.voltage.magnitude .= 1.0
    system.bus.voltage.angle .= 0.0
    JG.acModel!(system)
    return system
end

"""The join, for the Python side to assert: JuliaGrid keeps buses, branches
and generators in the `.m`'s row order (its labels are the bus names where
the case has `bus_name`, so they are no join), and refers to buses by index.
Returns the bus count, every branch's from and to bus index and every
generator's bus index, all 1-based."""
layout(system) = (system.bus.number, copy(system.branch.layout.from), copy(system.branch.layout.to),
                  copy(system.generator.layout.bus))

struct Model
    system::JG.PowerSystem
    analysis::JG.AcPowerFlow
end

"""Newton-Raphson on JuliaGrid's KLU factorization."""
function load(path::AbstractString)
    system = flat_system(path)
    return Model(system, JG.newtonRaphson(system, JG.KLU))
end

"""JuliaGrid's `powerFlow!` loop (a mismatch evaluation, a stop when both the
largest |ΔP| and |ΔQ| are below `tol`, at most `max_iterations` Newton
steps), returning the step count, or -1 if not converged: `powerFlow!`
itself reports neither."""
function iterate!(analysis::JG.AcPowerFlow, tol::Float64, max_iterations::Int)
    analysis.method.iteration = 0
    for _ in 0:max_iterations
        delP, delQ = JG.mismatch!(analysis)
        if delP < tol && delQ < tol
            return analysis.method.iteration
        end
        analysis.method.iteration == max_iterations && return -1
        JG.solve!(analysis)
    end
    return -1
end

"""The flat start (`setInitialPoint!`: the system's flat voltages, generator
setpoints at regulated buses), then the Newton loop."""
function solve!(m::Model, tol::Float64, max_iterations::Int)
    JG.setInitialPoint!(m.analysis)
    return iterate!(m.analysis, tol, max_iterations)
end

"""|V| in p.u. and angle in degrees, in bus order."""
solution(m::Model) = (copy(m.analysis.voltage.magnitude), rad2deg.(m.analysis.voltage.angle))

# ---------------------------------------------------------------- batch

"""
A sweep of operating points on one `Model`: per scenario the demand of the
loaded buses (`bus`, JuliaGrid bus indices) and the output of the online
generators (`gen`, JuliaGrid generator indices), p.u., scenarios in rows,
written through JuliaGrid's `updateBus!` and `updateGenerator!` on the
analysis; base values for the base case. Every scenario starts from the
base case's solution (`vm`, `va`).
"""
struct Sweep
    bus::Vector{Int}
    gen::Vector{Int}
    pd::Matrix{Float64}
    qd::Matrix{Float64}
    pg::Matrix{Float64}
    pd0::Vector{Float64}
    qd0::Vector{Float64}
    pg0::Vector{Float64}
    vm::Vector{Float64}
    va::Vector{Float64}
end

function load_sweep(m::Model, bus::AbstractVector, gen::AbstractVector, pd::AbstractMatrix, qd::AbstractMatrix,
                    pg::AbstractMatrix)
    system = m.system
    b, g = Int.(bus), Int.(gen)
    n = system.bus.number
    return Sweep(b, g, Matrix{Float64}(pd), Matrix{Float64}(qd), Matrix{Float64}(pg),
                 system.bus.demand.active[b], system.bus.demand.reactive[b], system.generator.output.active[g],
                 zeros(n), zeros(n))
end

"""Scenario k's demand and dispatch into the analysis; k = 0: the case's own."""
function apply!(m::Model, s::Sweep, k::Int)
    labels = collect(keys(m.system.bus.label))
    glabels = collect(keys(m.system.generator.label))
    for (j, i) in enumerate(s.bus)
        JG.updateBus!(m.analysis; label = labels[i], active = k == 0 ? s.pd0[j] : s.pd[k, j],
                      reactive = k == 0 ? s.qd0[j] : s.qd[k, j])
    end
    for (j, i) in enumerate(s.gen)
        JG.updateGenerator!(m.analysis; label = glabels[i], active = k == 0 ? s.pg0[j] : s.pg[k, j])
    end
end

"""The base case from the flat start; keeps its solution. Returns the
Newton steps, or -1."""
function solve_sweep_base!(m::Model, s::Sweep, tol::Float64, max_iterations::Int)
    apply!(m, s, 0)
    steps = solve!(m, tol, max_iterations)
    s.vm .= m.analysis.voltage.magnitude
    s.va .= m.analysis.voltage.angle
    return steps
end

"""Scenario k (1-based) from the base solution. Returns the Newton steps, or -1."""
function solve_scenario!(m::Model, s::Sweep, k::Int, tol::Float64, max_iterations::Int)
    apply!(m, s, k)
    m.analysis.voltage.magnitude .= s.vm
    m.analysis.voltage.angle .= s.va
    return iterate!(m.analysis, tol, max_iterations)
end

# ---------------------------------------------------------------- N-1

"""N-1 on one `Model`: `branches`, JuliaGrid branch indices (the `.m`'s
rows); each outage sets its branch's status to 0 through `updateBranch!`
on the analysis, starts from the base solution and puts the branch back."""
struct Outages
    branches::Vector{Int}
    vm::Vector{Float64}   # base solution, written by solve_n1_base!
    va::Vector{Float64}
end

load_outages(m::Model, branches::AbstractVector) = Outages(Int.(branches), zeros(m.system.bus.number), zeros(m.system.bus.number))

function solve_n1_base!(m::Model, o::Outages, tol::Float64, max_iterations::Int)
    steps = solve!(m, tol, max_iterations)
    o.vm .= m.analysis.voltage.magnitude
    o.va .= m.analysis.voltage.angle
    return steps
end

function solve_outage!(m::Model, o::Outages, k::Int, tol::Float64, max_iterations::Int)
    label = collect(keys(m.system.branch.label))[o.branches[k]]
    JG.updateBranch!(m.analysis; label = label, status = 0)
    m.analysis.voltage.magnitude .= o.vm
    m.analysis.voltage.angle .= o.va
    steps = iterate!(m.analysis, tol, max_iterations)
    JG.updateBranch!(m.analysis; label = label, status = 1)
    return steps
end

# ---------------------------------------------------------------- state estimation

struct Estimator
    system::JG.PowerSystem
    analysis::JG.AcStateEstimation
    tol::Float64
    max_iterations::Int
end

"""WLS state estimation, Gauss-Newton on the normal equations with
JuliaGrid's KLU. Measurements, one per row: `kind` 0 |V| (bus), 1 P and 2 Q
injection (bus), 3 P and 4 Q from-end flow (branch, JuliaGrid index), value
and sigma in p.u.; the variance JuliaGrid takes is sigma²."""
function load_se(path::AbstractString, kind::AbstractVector, bus::AbstractVector, branch::AbstractVector,
                 value::AbstractVector, sigma::AbstractVector, tol::Float64, max_iterations::Int)
    system = flat_system(path)
    monitoring = JG.measurement(system)
    labels = collect(keys(system.bus.label))
    blabels = collect(keys(system.branch.label))
    for (k, b, br, v, s) in zip(kind, bus, branch, value, sigma)
        var = s^2
        if k == 0
            JG.addVoltmeter!(monitoring; bus = labels[b], magnitude = v, variance = var)
        elseif k == 1
            JG.addWattmeter!(monitoring; bus = labels[b], active = v, variance = var)
        elseif k == 2
            JG.addVarmeter!(monitoring; bus = labels[b], reactive = v, variance = var)
        elseif k == 3
            JG.addWattmeter!(monitoring; from = blabels[br], active = v, variance = var)
        elseif k == 4
            JG.addVarmeter!(monitoring; from = blabels[br], reactive = v, variance = var)
        else
            error("unknown measurement kind $k")
        end
    end
    return Estimator(system, JG.gaussNewton(monitoring, JG.KLU), tol, max_iterations)
end

"""From the flat start (`setInitialPoint!`), JuliaGrid's `stateEstimation!`
loop: stop when the largest state increment is below `tol`, at most
`max_iterations` Gauss-Newton steps. Returns the steps, or -1."""
function estimate!(e::Estimator)
    analysis = e.analysis
    JG.setInitialPoint!(analysis)
    analysis.method.iteration = 0
    for _ in 0:e.max_iterations
        JG.increment!(analysis) < e.tol && return analysis.method.iteration
        analysis.method.iteration == e.max_iterations && return -1
        JG.solve!(analysis)
    end
    return -1
end

se_solution(e::Estimator) = (copy(e.analysis.voltage.magnitude), rad2deg.(e.analysis.voltage.angle))

# ---------------------------------------------------------------- AC optimal power flow

struct Optimizer
    system::JG.PowerSystem
    analysis::JG.AcOptimalPowerFlow
end

"""MATPOWER's AC-OPF as JuliaGrid builds it (`acOptimalPowerFlow`) with
Ipopt: `tol` and `max_iter` from the caller, MUMPS (Ipopt_jll's), silent.
Start: every bus at 1 p.u. and 0 rad, every generator at the middle of its P
and Q ranges, written into the system so that `setInitialPoint!` restores
it (and drops the previous solve's duals) before every solve."""
function load_opf(path::AbstractString, tol::Float64, max_iterations::Int)
    system = flat_system(path; optimal = true)
    gen = system.generator
    gen.output.active .= (gen.capability.minActive .+ gen.capability.maxActive) ./ 2
    gen.output.reactive .= (gen.capability.minReactive .+ gen.capability.maxReactive) ./ 2
    analysis = JG.acOptimalPowerFlow(system, Ipopt.Optimizer)
    jump = analysis.method.jump
    JuMP.set_silent(jump)
    JuMP.set_attribute(jump, "tol", tol)
    JuMP.set_attribute(jump, "max_iter", max_iterations)
    JuMP.set_attribute(jump, "linear_solver", "mumps")
    return Optimizer(system, analysis)
end

"""Restores the start, solves. Returns (locally solved, Ipopt iterations)."""
function solve_opf!(o::Optimizer)
    JG.setInitialPoint!(o.analysis)
    JG.solve!(o.analysis)
    jump = o.analysis.method.jump
    ok = JuMP.termination_status(jump) == JuMP.LOCALLY_SOLVED
    return ok, Int(JuMP.MOI.get(jump, JuMP.MOI.BarrierIterations()))
end

"""|V| p.u. and angle degrees in bus order; per generator (the gen rows) P and
Q in p.u.; the objective as JuliaGrid states it."""
opf_solution(o::Optimizer) = (copy(o.analysis.voltage.magnitude),
                              rad2deg.(o.analysis.voltage.angle), copy(o.analysis.power.generator.active),
                              copy(o.analysis.power.generator.reactive),
                              JuMP.objective_value(o.analysis.method.jump))

versions() = Dict("JuliaGrid" => pkgversion(JG), "JuMP" => pkgversion(JuMP), "Ipopt" => pkgversion(Ipopt))

# Info and warning logs off (JuliaGrid logs a moved slack bus at info level,
# which would be timed). Errors still print.
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
        Logging.with_logger(Logging.NullLogger()) do
            redirect_stdout(devnull) do
                m = load(path)
                solve!(m, 1e-8, 30)
                solution(m)
                layout(m.system)
                s = load_sweep(m, [3], [2], reshape([0.8, 0.2], 2, 1), reshape([0.25, 0.05], 2, 1), reshape([0.4, 0.6], 2, 1))
                solve_sweep_base!(m, s, 1e-8, 30)
                solve_scenario!(m, s, 2, 1e-8, 30)
                o = load_outages(m, [1])
                solve_n1_base!(m, o, 1e-8, 30)
                solve_outage!(m, o, 1, 1e-8, 30)
                e = load_se(path, [0, 1, 2, 1, 2, 3, 4], [1, 2, 2, 3, 3, 0, 0], [0, 0, 0, 0, 0, 1, 1],
                            [1.02, 0.5, 0.1, -0.9, -0.3, 0.2, 0.05], [0.004, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                            1e-8, 30)
                estimate!(e)
                se_solution(e)
                op = load_opf(path, 1e-6, 200)
                solve_opf!(op)
                opf_solution(op)
            end
        end
    end
end

end
