"""
The Julia half of adapters/exapf_adapter.py (settings and their
justification are documented there). It lives in the image, not in
adapters/, because it is compiled into it: the workload at the bottom runs
load, solve and solution on a 3-bus case at build time, so every fresh
process starts with native code instead of spending its first case in the
JIT (see GridBenchSienna). Changing this file needs `docker/build.sh exapf`.

It depends on PythonCall (unused here) because juliacall loads PythonCall
first: compiled without it, the cached code is invalidated by PythonCall's
method definitions and recompiled on first use.
"""
module GridBenchExaPF

using ExaPF
using KLU: KLU
using Logging
using PrecompileTools: @compile_workload, @setup_workload
using PythonCall: PythonCall

const PS = ExaPF.PowerSystem
const LS = ExaPF.LinearSolvers

struct Model
    prob::ExaPF.PowerFlowProblem
    vm0::Vector{Float64}   # flat start: slack and PV buses at their setpoints, every other bus at 1 p.u.
end

"""ExaPF's documented entry point for a single power flow, on the CPU
backend. `max_iterations + 1`: ExaPF evaluates the mismatch at the top of
each of its `maxiter` passes and breaks there, so the last pass can only
check, never step; this allows exactly `max_iterations` Newton steps, each
followed by a check."""
function load(path::AbstractString, tol::Float64, max_iterations::Int)
    prob = ExaPF.PowerFlowProblem(String(path), ExaPF.CPU(), :polar; rtol = tol, max_iter = max_iterations + 1)
    @assert prob.linear_solver isa LS.DirectSolver{<:KLU.KLUFactorization}
    net = prob.form.network
    vm0 = ones(net.nbus)
    regulated = [net.ref; net.pv]
    vm0[regulated] .= prob.stack.vmag[regulated]   # init!: the generator's Vg on every bus ExaPF regulates
    return Model(prob, vm0)
end

"""Restores the flat start (the stack holds the previous solution), then
solves. Returns (converged, Newton steps)."""
function solve!(m::Model)
    m.prob.stack.vmag .= m.vm0
    m.prob.stack.vang .= 0.0
    conv = ExaPF.solve!(m.prob)
    return conv.has_converged, conv.n_iterations
end

"""Bus numbers (MATPOWER's), |V| in p.u., angle in degrees. ExaPF indexes
buses by their row in the `.m`'s bus matrix (`bus_to_indexes`), asserted."""
function solution(m::Model)
    net = m.prob.form.network
    numbers = Int.(net.buses[:, 1])
    @assert all(net.bus_to_indexes[n] == i for (i, n) in enumerate(numbers))
    return numbers, copy(m.prob.stack.vmag), rad2deg.(m.prob.stack.vang)
end

"""
A batch: ExaPF's `BlockPolarForm`, k copies of the network solved as one
block-diagonal Newton system (`BatchJacobian`, one KLU of the whole block
diagonal). The construction is `PowerFlowProblem(..., :block_polar, ...)`'s,
spelled out: that constructor needs the loads of every bus before the
network is parsed, so using it would parse the `.m` twice. Scenario data in
p.u. on the case's base power, scenarios in rows: `pd`, `qd` per bus
(`load_bus`, MATPOWER numbers, joined by `bus_to_indexes`), `pg` per online
generator (`gen_pos`, 1-based among ExaPF's online generators, which are the
`.m`'s in file order; asserted against `gen_bus`). Every other load and
dispatch stays the case's.
"""
struct Batch
    stack::ExaPF.NetworkStack
    jac::ExaPF.BatchJacobian
    linear_solver::LS.DirectSolver
    algo::ExaPF.NewtonRaphson
    buffer::ExaPF.NLBuffer
    vm0::Vector{Float64}
    numbers::Vector{Int}
    vm::Matrix{Float64}   # buses x scenarios, written by solve_batch!
    va::Matrix{Float64}
end

function load_batch(path::AbstractString, tol::Float64, max_iterations::Int,
                    load_bus::AbstractVector, pd::AbstractMatrix, qd::AbstractMatrix,
                    gen_pos::AbstractVector, gen_bus::AbstractVector, pg::AbstractMatrix)
    polar = ExaPF.PolarForm(String(path), ExaPF.CPU())
    net = polar.network
    k = size(pd, 1)
    blk = ExaPF.BlockPolarForm(polar, k)
    stack = ExaPF.NetworkStack(blk)
    rows = [net.bus_to_indexes[Int(b)] for b in load_bus]
    reshape(stack.pload, net.nbus, k)[rows, :] .= permutedims(pd)
    reshape(stack.qload, net.nbus, k)[rows, :] .= permutedims(qd)
    gens = Int.(gen_pos)
    @assert Int.(net.generators[gens, 1]) == Int.(gen_bus)
    reshape(stack.pgen, net.ngen, k)[gens, :] .= permutedims(pg)

    powerflow = ExaPF.PowerFlowBalance(blk) ∘ ExaPF.Basis(blk)
    jac = ExaPF.BatchJacobian(blk, powerflow, ExaPF.State())
    ExaPF.set_params!(jac, stack)
    ExaPF.jacobian!(jac, stack)
    linear_solver = ExaPF.default_linear_solver(jac.J; nblocks = k)
    @assert linear_solver isa LS.DirectSolver{<:KLU.KLUFactorization}
    vm0 = ones(net.nbus)
    regulated = [net.ref; net.pv]
    vm0[regulated] .= stack.vmag[regulated]
    numbers = Int.(net.buses[:, 1])
    @assert all(net.bus_to_indexes[n] == i for (i, n) in enumerate(numbers))
    return Batch(stack, jac, linear_solver, ExaPF.NewtonRaphson(tol = tol, maxiter = max_iterations + 1),
                 ExaPF.NLBuffer{Vector{Float64}}(size(jac.J, 2)), repeat(vm0, k), numbers,
                 zeros(net.nbus, k), zeros(net.nbus, k))
end

"""Restores every scenario's flat start, solves them all, keeps every
scenario's voltages. Returns (converged, scenarios whose own mismatch
2-norm is not below `tol`): ExaPF's test is the 2-norm over all of them."""
function solve_batch!(b::Batch)
    b.stack.vmag .= b.vm0
    b.stack.vang .= 0.0
    conv = ExaPF.nlsolve!(b.algo, b.jac, b.stack; linear_solver = b.linear_solver, nl_buffer = b.buffer)
    nbus, k = size(b.vm)
    b.vm .= reshape(b.stack.vmag, nbus, k)
    b.va .= rad2deg.(reshape(b.stack.vang, nbus, k))
    conv.has_converged && return true, 0
    per_block = reshape(b.buffer.y, :, k)   # nlsolve! leaves the final mismatch here, scenario by scenario
    return false, count(j -> !(sqrt(sum(abs2, view(per_block, :, j))) < b.algo.tol), 1:k)
end

"""Bus numbers, |V| in p.u. and angle in degrees, buses x scenarios, of the last solve_batch!."""
batch_solution(b::Batch) = (b.numbers, b.vm, b.va)

"""
N-1: ExaPF's line-contingency formulation, `PowerFlowBalance(blk,
contingencies)`: block 1 the base case, block 1 + j the network with
contingency j's line admittances zeroed (`PS.drop_line`), one
block-diagonal Newton system on a `BatchJacobian`. A `LineContingency` is
the 1-based row of the `.m`'s branch matrix (out-of-service rows
included), asserted against each outage's buses. `base` is the single power
flow (`load`), whose network the blocks reuse, so the `.m` is parsed once.
"""
struct N1
    base::Model
    stack::ExaPF.NetworkStack
    jac::ExaPF.BatchJacobian
    linear_solver::LS.DirectSolver
    algo::ExaPF.NewtonRaphson
    buffer::ExaPF.NLBuffer
    vm::Matrix{Float64}   # buses x outages, written by solve_n1!
    va::Matrix{Float64}
end

"""A lower bound on the memory of k blocks, in bytes: the `BatchJacobian`'s
own `NetworkStack` of ForwardDiff duals (one partial per colour of the
Jacobian, the single Jacobian's colouring, which `BatchJacobian` repeats per
block), counted from ExaPF's layout (`NetworkStack`: input 2nbus + ngen,
basis 2nlines + nbus, intermediates ngen + 8 nlines). For 100 blocks of
case1354pegase it counts 0.51 GB; the stack measured 0.55 GB of 0.72 GB
live."""
function dual_bytes(m::Model, k::Int)
    net = m.prob.form.network
    nlines = size(net.branches, 1)
    per_block = 3 * net.nbus + 2 * net.ngen + 10 * nlines
    return k * per_block * (m.prob.jac.ncolors + 1) * sizeof(Float64)
end

function load_n1(base::Model, rows::AbstractVector, from::AbstractVector, to::AbstractVector)
    polar = base.prob.form
    net = polar.network
    lines = [Int(r) + 1 for r in rows]
    @assert Int.(net.branches[lines, 1]) == Int.(from) && Int.(net.branches[lines, 2]) == Int.(to)
    k = length(lines) + 1
    blk = ExaPF.BlockPolarForm(polar, k)
    stack = ExaPF.NetworkStack(blk)
    powerflow = ExaPF.PowerFlowBalance(blk, ExaPF.LineContingency.(lines)) ∘ ExaPF.Basis(blk)
    jac = ExaPF.BatchJacobian(blk, powerflow, ExaPF.State())
    ExaPF.set_params!(jac, stack)
    ExaPF.jacobian!(jac, stack)
    linear_solver = ExaPF.default_linear_solver(jac.J; nblocks = k)
    @assert linear_solver isa LS.DirectSolver{<:KLU.KLUFactorization}
    return N1(base, stack, jac, linear_solver, base.prob.non_linear_solver,
              ExaPF.NLBuffer{Vector{Float64}}(size(jac.J, 2)), zeros(net.nbus, k - 1), zeros(net.nbus, k - 1))
end

"""The base case from flat start (the single power flow), then every block
from its solution. Returns (base converged, outages whose own mismatch
2-norm is not below the tolerance)."""
function solve_n1!(n::N1)
    converged, _ = solve!(n.base)
    converged || return false, 0
    nbus, k = size(n.vm, 1), size(n.vm, 2) + 1
    reshape(n.stack.vmag, nbus, k) .= n.base.prob.stack.vmag
    reshape(n.stack.vang, nbus, k) .= n.base.prob.stack.vang
    conv = ExaPF.nlsolve!(n.algo, n.jac, n.stack; linear_solver = n.linear_solver, nl_buffer = n.buffer)
    n.vm .= view(reshape(n.stack.vmag, nbus, k), :, 2:k)
    n.va .= rad2deg.(view(reshape(n.stack.vang, nbus, k), :, 2:k))
    conv.has_converged && return true, 0
    per_block = reshape(n.buffer.y, :, k)
    return true, count(j -> !(sqrt(sum(abs2, view(per_block, :, j))) < n.algo.tol), 2:k)
end

"""Bus numbers, |V| in p.u. and angle in degrees, buses x outages, of the last solve_n1!."""
n1_solution(n::N1) = (solution(n.base)[1], n.vm, n.va)

versions() = Dict("ExaPF" => pkgversion(ExaPF), "KLU" => pkgversion(KLU))

# Info and warning logs off: ExaPF's parser logs at info level on every
# field outside MATPOWER's core matrices, which would be timed. Errors still print.
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
    mpc.gencost = [
        2 0 0 3 0 1 0;
        2 0 0 3 0 1 0;
    ];
    """
    path = joinpath(mktempdir(), "gb3.m")
    write(path, case)
    @compile_workload begin
        Logging.with_logger(Logging.NullLogger()) do
            m = load(path, 1e-8, 30)
            solve!(m)
            solution(m)
            b = load_batch(path, 1e-8, 30, [3], [0.9; 0.8;;], [0.3; 0.25;;], [2], [2], [0.5; 0.4;;])
            solve_batch!(b)
            batch_solution(b)
            o = load_n1(load(path, 1e-8, 30), [0, 2], [1, 1], [2, 3])
            dual_bytes(o.base, 3)
            solve_n1!(o)
            n1_solution(o)
        end
    end
end

end
