"""
The Julia half of adapters/exapf_adapter.py and adapters/exapf_gpu_adapter.py
(settings and their justification are documented there). It lives in the
image, not in adapters/, because it is compiled into it: the workload at the
bottom runs load, solve and solution on a 3-bus case at build time, so every
fresh process starts with native code instead of spending its first case in
the JIT (see GridBenchSienna). Changing this file needs
`docker/build.sh exapf exapf_gpu`.

Every entry point takes the KernelAbstractions backend (`ExaPF.CPU()` or
CUDA's `CUDABackend()`) and the factorization the linear solver must use
(KLU's or cuDSS's), both from the caller: this package does not depend on
CUDA, so the CPU image does not carry it. ExaPF keeps a model's arrays on the
backend's device; data goes in through `copyto!` and results come out
through `Array`, plain copies on the CPU. The precompile workload runs on the
CPU only, as the images are built without a GPU: CUDA kernels compile on a
case's untimed warm-up instead.

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

struct Model{VT}
    prob::ExaPF.PowerFlowProblem
    vm0::VT   # flat start, on the device: slack and PV buses at their setpoints, every other bus at 1 p.u.
end

"""A copy of host vector `x` with the array type (and device) of `like`."""
on_device(like::AbstractVector, x::AbstractVector) = copyto!(similar(like, length(x)), x)

"""Waits for the device: a GPU runs kernels asynchronously, and a timed call
must return only once its work is done. A no-op on the CPU."""
sync(backend) = ExaPF.KA.synchronize(backend)

"""The flat-start magnitudes of `net`'s buses: the generator's Vg (from
`vmag`, where ExaPF's `init!` puts it) on every bus ExaPF regulates, 1 p.u.
elsewhere."""
function flat_vm(net, vmag::AbstractVector)
    vm0 = ones(net.nbus)
    regulated = [net.ref; net.pv]
    vm0[regulated] .= Array(vmag)[regulated]
    return vm0
end

"""ExaPF's documented entry point for a single power flow, on `backend`.
`max_iterations + 1`: ExaPF evaluates the mismatch at the top of
each of its `maxiter` passes and breaks there, so the last pass can only
check, never step; this allows exactly `max_iterations` Newton steps, each
followed by a check."""
function load(path::AbstractString, backend, factorization::Type, tol::Float64, max_iterations::Int)
    prob = ExaPF.PowerFlowProblem(String(path), backend, :polar; rtol = tol, max_iter = max_iterations + 1)
    @assert prob.linear_solver isa LS.DirectSolver{<:factorization}
    return Model(prob, on_device(prob.stack.vmag, flat_vm(prob.form.network, prob.stack.vmag)))
end

"""Restores the flat start (the stack holds the previous solution), then
solves. Returns (converged, Newton steps)."""
function solve!(m::Model)
    m.prob.stack.vmag .= m.vm0
    m.prob.stack.vang .= 0.0
    conv = ExaPF.solve!(m.prob)
    sync(m.prob.backend)
    return conv.has_converged, conv.n_iterations
end

"""Bus numbers (MATPOWER's), |V| in p.u., angle in degrees. ExaPF indexes
buses by their row in the `.m`'s bus matrix (`bus_to_indexes`), asserted."""
function solution(m::Model)
    net = m.prob.form.network
    numbers = Int.(net.buses[:, 1])
    @assert all(net.bus_to_indexes[n] == i for (i, n) in enumerate(numbers))
    return numbers, Array(m.prob.stack.vmag), rad2deg.(Array(m.prob.stack.vang))
end

"""
A batch: ExaPF's `BlockPolarForm`, k copies of the network solved as one
block-diagonal Newton system (`BatchJacobian`; one KLU of the whole block
diagonal on the CPU, one uniform batch of cuDSS on a GPU). The construction
is `PowerFlowProblem(..., :block_polar, ...)`'s,
spelled out: that constructor needs the loads of every bus before the
network is parsed, so using it would parse the `.m` twice. Scenario data in
p.u. on the case's base power, scenarios in rows: `pd`, `qd` per bus
(`load_bus`, MATPOWER numbers, joined by `bus_to_indexes`), `pg` per online
generator (`gen_pos`, 1-based among ExaPF's online generators, which are the
`.m`'s in file order; asserted against `gen_bus`). Every other load and
dispatch stays the case's. `base` is the single power flow of the case
(`load`), whose solution every scenario starts from.
"""
struct Batch{M <: Model, BT}
    base::M
    backend::Any
    stack::ExaPF.NetworkStack
    jac::ExaPF.BatchJacobian
    linear_solver::LS.DirectSolver
    algo::ExaPF.NewtonRaphson
    buffer::ExaPF.NLBuffer{BT}
    numbers::Vector{Int}
    vm::Matrix{Float64}   # buses x scenarios, on the host, written by solve_batch!
    va::Matrix{Float64}
end

"""Writes `values` (scenarios in rows) into rows `rows` of every block of
device vector `x` (`n` entries per block); every other row keeps its value."""
function set_blocks!(x::AbstractVector, n::Int, rows::AbstractVector{Int}, values::AbstractMatrix)
    host = reshape(Array(x), n, :)
    host[rows, :] .= permutedims(values)
    copyto!(x, vec(host))
end

function load_batch(path::AbstractString, backend, factorization::Type, tol::Float64, max_iterations::Int,
                    load_bus::AbstractVector, pd::AbstractMatrix, qd::AbstractMatrix,
                    gen_pos::AbstractVector, gen_bus::AbstractVector, pg::AbstractMatrix)
    polar = ExaPF.PolarForm(String(path), backend)
    net = polar.network
    k = size(pd, 1)
    blk = ExaPF.BlockPolarForm(polar, k)
    stack = ExaPF.NetworkStack(blk)
    rows = [net.bus_to_indexes[Int(b)] for b in load_bus]
    set_blocks!(stack.pload, net.nbus, rows, pd)
    set_blocks!(stack.qload, net.nbus, rows, qd)
    gens = Int.(gen_pos)
    @assert Int.(net.generators[gens, 1]) == Int.(gen_bus)
    set_blocks!(stack.pgen, net.ngen, gens, pg)

    powerflow = ExaPF.PowerFlowBalance(blk) ∘ ExaPF.Basis(blk)
    jac = ExaPF.BatchJacobian(blk, powerflow, ExaPF.State())
    ExaPF.set_params!(jac, stack)
    ExaPF.jacobian!(jac, stack)
    linear_solver = ExaPF.default_linear_solver(jac.J; nblocks = k)
    @assert linear_solver isa LS.DirectSolver{<:factorization}
    numbers = Int.(net.buses[:, 1])
    @assert all(net.bus_to_indexes[n] == i for (i, n) in enumerate(numbers))
    base = load(path, backend, factorization, tol, max_iterations)
    @assert Int.(base.prob.form.network.buses[:, 1]) == numbers
    return Batch(base, backend, stack, jac, linear_solver, ExaPF.NewtonRaphson(tol = tol, maxiter = max_iterations + 1),
                 ExaPF.NLBuffer{typeof(stack.params)}(size(jac.J, 2)), numbers, zeros(net.nbus, k), zeros(net.nbus, k))
end

"""Copies the voltages of `stack`'s blocks from block `first` on into the
host matrices `vm` (p.u.) and `va` (degrees), buses x blocks."""
function copy_blocks!(vm::Matrix{Float64}, va::Matrix{Float64}, stack::ExaPF.NetworkStack, first::Int)
    offset = (first - 1) * size(vm, 1) + 1
    copyto!(vm, 1, stack.vmag, offset, length(vm))
    copyto!(va, 1, stack.vang, offset, length(va))
    va .= rad2deg.(va)
end

"""How many of `blocks` have a mismatch 2-norm of their own not below `tol`:
`nlsolve!` leaves the final mismatch in `buffer.y`, block after block."""
function unconverged(buffer::ExaPF.NLBuffer, k::Int, blocks, tol::Float64)
    per_block = reshape(Array(buffer.y), :, k)
    return count(j -> !(sqrt(sum(abs2, view(per_block, :, j))) < tol), blocks)
end

"""The base case from flat start (the single power flow), then every
scenario from its solution, solved all at once; copies every scenario's
voltages to the host. Returns (base converged, scenarios whose own mismatch
2-norm is not below `tol`): ExaPF's test is the 2-norm over all of them."""
function solve_batch!(b::Batch)
    converged, _ = solve!(b.base)
    converged || return false, 0
    nbus, k = size(b.vm)
    reshape(b.stack.vmag, nbus, k) .= b.base.prob.stack.vmag
    reshape(b.stack.vang, nbus, k) .= b.base.prob.stack.vang
    conv = ExaPF.nlsolve!(b.algo, b.jac, b.stack; linear_solver = b.linear_solver, nl_buffer = b.buffer)
    sync(b.backend)
    copy_blocks!(b.vm, b.va, b.stack, 1)
    conv.has_converged && return true, 0
    return true, unconverged(b.buffer, k, 1:k, b.algo.tol)
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
flow (`load`), whose network and backend the blocks reuse, so the `.m` is
parsed once; the blocks' linear solver is asserted to be the base case's.
"""
struct N1{M <: Model, BT}
    base::M
    stack::ExaPF.NetworkStack
    jac::ExaPF.BatchJacobian
    linear_solver::LS.DirectSolver
    algo::ExaPF.NewtonRaphson
    buffer::ExaPF.NLBuffer{BT}
    vm::Matrix{Float64}   # buses x outages, on the host, written by solve_n1!
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
    @assert typeof(linear_solver) == typeof(base.prob.linear_solver)
    return N1(base, stack, jac, linear_solver, base.prob.non_linear_solver,
              ExaPF.NLBuffer{typeof(stack.params)}(size(jac.J, 2)), zeros(net.nbus, k - 1), zeros(net.nbus, k - 1))
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
    sync(n.base.prob.backend)
    copy_blocks!(n.vm, n.va, n.stack, 2)
    conv.has_converged && return true, 0
    return true, unconverged(n.buffer, k, 2:k, n.algo.tol)
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
            cpu, klu = ExaPF.CPU(), KLU.KLUFactorization
            m = load(path, cpu, klu, 1e-8, 30)
            solve!(m)
            solution(m)
            b = load_batch(path, cpu, klu, 1e-8, 30, [3], [0.9; 0.8;;], [0.3; 0.25;;], [2], [2], [0.5; 0.4;;])
            solve_batch!(b)
            batch_solution(b)
            o = load_n1(load(path, cpu, klu, 1e-8, 30), [0, 2], [1, 1], [2, 3])
            dual_bytes(o.base, 3)
            solve_n1!(o)
            n1_solution(o)
        end
    end
end

end
