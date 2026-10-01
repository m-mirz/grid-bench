"""
The Julia half of adapters/sparlectra_adapter.py (settings and their
justification are documented there). It lives in the image, not in
adapters/, because it is compiled into it: the workload at the bottom runs
load, solve and solution (and the state-estimation half, se.jl) on a 3-bus MATPOWER case and on Sparlectra's own
CGMES demo (case14, without its SV) at build time, so every fresh process
starts with native code instead of spending its first case in the JIT.
Changing this file needs `docker/build.sh sparlectra`.

It depends on PythonCall (unused here) for the same reason as
GridBenchSienna: juliacall loads PythonCall first, and code compiled without
it is invalidated by PythonCall's method definitions.
"""
module GridBenchSparlectra

using Logging
using PrecompileTools: @compile_workload, @setup_workload
using PythonCall: PythonCall
using Sparlectra

const SP = Sparlectra

struct Model
    net::SP.Net
    vm0::Vector{Union{Nothing,Float64}}   # imported start state, restored before every solve
    va0::Vector{Union{Nothing,Float64}}
    types0::Vector{SP.NodeType}           # bus types as imported, restored before every solve
    nodes::Vector{Int}                    # node index of each id
    ids::Vector{String}                   # MATPOWER bus number, or TopologicalNode mRID
    vn_kV::Vector{Float64}                # 1.0 for MATPOWER (p.u. out), nominal kV for CGMES
end

function Model(net::SP.Net, nodes::Vector{Int}, ids::Vector{String}, vn_kV::Vector{Float64})
    # Isolated buses (MATPOWER type 4, CGMES buses without an energised
    # branch) are excluded from the solve and so from the solution.
    keep = [SP.getNodeType(net.nodeVec[k]) != SP.Isolated for k in nodes]
    # The common flat start, as MATPOWER's runpf and the other adapters use
    # it: PV and slack buses at their generators' voltage setpoints, PQ buses
    # at 1 p.u., every angle 0. Sparlectra's own flat start would take PV
    # magnitudes and the slack angle from the node, i.e. from the case's bus
    # VM/VA columns, which are not setpoints (and on PEGASE not a solution).
    vset = SP._bus_voltage_setpoints_from_prosumers(net)
    vm0 = Union{Nothing,Float64}[SP.getNodeType(n) in (SP.PV, SP.Slack) ? vset[k] : 1.0 for (k, n) in enumerate(net.nodeVec)]
    return Model(net, vm0, zeros(length(vm0)), [n._nodeType for n in net.nodeVec], nodes[keep], ids[keep], vn_kV[keep])
end

function load_matpower(path::AbstractString)
    net = SP.createNetFromMatPowerFile(;
        filename = String(path),
        flatstart = true,
        enable_pq_gen_controllers = false,
        bus_shunt_model = :admittance,
        matpower_shift_sign = 1.0,
        matpower_shift_unit = :deg,
        matpower_ratio = :normal,
        tap_changer_model = :ideal,
        matpower_pv_voltage_source = :gen_vg,
        matpower_dcline_mode = :pf_injections,
    )
    # Explicit join: addBus! records each node's MATPOWER bus number.
    n = length(net.nodeVec)
    @assert length(net.busOrigIdxDict) == n "not every node carries its MATPOWER bus number"
    nodes = collect(1:n)
    ids = [string(net.busOrigIdxDict[k]) for k in nodes]
    @assert allunique(ids) "MATPOWER bus numbers are not unique across nodes"
    return Model(net, nodes, ids, ones(n))
end

function load_cgmes(path)   # a file, zip or directory, or a vector of them
    result = SP.importCGMES(;
        path = path,
        baseMVA = 100.0,
        bus_shunt_model = :admittance,
        require_boundary = true,
        tap_control = false,
        machine_control = false,
        multi_slack = false,
        hvdc_mode = :injections,
    )
    net = result.net
    # Explicit join: the importer names each bus after its TopologicalNode
    # (topo.bus_name); per-side boundary buses and auxiliary buses have no TN
    # of their own and are left out.
    tns = [tn for (tn, bus) in result.topo.bus_name if haskey(net.busDict, bus)]
    nodes = [net.busDict[result.topo.bus_name[tn]] for tn in tns]
    @assert allunique(nodes) "two TopologicalNodes map onto one bus"
    return Model(net, nodes, tns, [result.topo.vn_kV[tn] for tn in tns])
end

"""Restores the flat start (Sparlectra's flat start reads PV and slack
magnitudes and the slack angle from the node, which the previous solve
overwrote) and the imported bus types (a solve writes its internal types
back, turning every Isolated bus into PQ, so a second solve on the same Net
would include unconnected buses and diverge), then solves. The loop counts mismatch evaluations, so
`max_iterations + 1` allows exactly `max_iterations` Newton steps. Returns
the number of Newton steps, or -1 if not converged."""
function solve!(m::Model, tol::Float64, max_iterations::Int)
    for (k, node) in enumerate(m.net.nodeVec)
        node._vm_pu = m.vm0[k]
        node._va_deg = m.va0[k]
        node._nodeType = m.types0[k]
    end
    return runpf!(m.net, tol, max_iterations, true)
end

"""The solve settings (see adapters/sparlectra_adapter.py); `flatstart`
false starts from the voltages held in the nodes."""
function runpf!(net::SP.Net, tol::Float64, max_iterations::Int, flatstart::Bool)
    iters, status = SP.runpf_rectangular!(
        net;
        method = :rectangular,
        maxiter = max_iterations + 1,
        tol = tol,
        damp = 1.0,
        verbose = 0,
        autodamp = false,
        merit_enabled = false,
        trust_region_enabled = false,
        opt_flatstart = flatstart,
        start_projection = false,
        start_current_iteration_enabled = false,
        apslf_start_enabled = false,
        qlimits_enabled = false,
        wrong_branch_detection = :off,
        wrong_branch_rescue = false,
        distributed_slack_enabled = false,
    )
    return status == 0 ? iters - 1 : -1
end

"""Ids, |V| (p.u. for MATPOWER, kV for CGMES) and angle in degrees."""
function solution(m::Model)
    nodes = m.net.nodeVec
    vm = [nodes[k]._vm_pu * vn for (k, vn) in zip(m.nodes, m.vn_kV)]
    va = [nodes[k]._va_deg for k in m.nodes]
    return m.ids, vm, va
end

"""
A sweep of operating points on one MATPOWER `Model`, for the batch
benchmark (adapters/sparlectra_batch_adapter.py). Sparlectra's solver builds
its injections from the prosumers (`pVal`, `qVal`, MW and MVAr) on every
solve, so scenario k sets the load prosumer of every loaded bus and one
generator prosumer at every generator bus to its imported value plus that
scenario's change: P and Q of the load, and the summed change of the bus's
online generators' P (at a PV bus only the total P enters the equations).
`numbers`: MATPOWER bus numbers of the rows of the change matrices (buses x
scenarios). Joined through `busOrigIdxDict`; a loaded bus must have exactly
one load prosumer, a generator bus with a change at least one generator.
"""
struct Sweep
    loads::Vector{Int}      # prosumer index per row, 0: none
    gens::Vector{Int}
    p_load0::Vector{Float64}
    q_load0::Vector{Float64}
    p_gen0::Vector{Float64}
    dp_load::Matrix{Float64}
    dq_load::Matrix{Float64}
    dp_gen::Matrix{Float64}
end

function load_sweep(m::Model, numbers::AbstractVector, dp_load::AbstractMatrix, dq_load::AbstractMatrix,
                    dp_gen::AbstractMatrix)
    node_of = Dict(Int(n) => k for (k, n) in m.net.busOrigIdxDict)
    ps = m.net.prosumpsVec
    bus(p) = Int(p.comp.cFrom_bus)
    is_load(p) = p.proSumptionType == SP.Consumption
    loads, gens = zeros(Int, length(numbers)), zeros(Int, length(numbers))
    for (i, b) in enumerate(numbers)
        node = node_of[Int(b)]
        at_load = [j for (j, p) in enumerate(ps) if bus(p) == node && is_load(p)]
        at_gen = [j for (j, p) in enumerate(ps) if bus(p) == node && !is_load(p)]
        if any(!=(0.0), dp_load[i, :]) || any(!=(0.0), dq_load[i, :])
            @assert length(at_load) == 1 "bus $b: $(length(at_load)) load prosumers"
            loads[i] = at_load[1]
        end
        if any(!=(0.0), dp_gen[i, :])
            @assert !isempty(at_gen) "bus $b: no generator prosumer"
            gens[i] = at_gen[1]
        end
    end
    val(j, f) = j == 0 ? 0.0 : something(getfield(ps[j], f), 0.0)
    return Sweep(loads, gens, [val(j, :pVal) for j in loads], [val(j, :qVal) for j in loads],
                 [val(j, :pVal) for j in gens], Matrix{Float64}(dp_load), Matrix{Float64}(dq_load),
                 Matrix{Float64}(dp_gen))
end

function apply!(m::Model, s::Sweep, k::Int)
    ps = m.net.prosumpsVec
    for i in eachindex(s.loads)
        if s.loads[i] != 0
            ps[s.loads[i]].pVal = s.p_load0[i] + s.dp_load[i, k]
            ps[s.loads[i]].qVal = s.q_load0[i] + s.dq_load[i, k]
        end
        if s.gens[i] != 0
            ps[s.gens[i]].pVal = s.p_gen0[i] + s.dp_gen[i, k]
        end
    end
end

"""
N-1 on one MATPOWER `Model`, for the contingency benchmark
(adapters/sparlectra_n1_adapter.py): the base case from the common flat
start, then every outage with its branch's status (and both end statuses)
set to 0, as Sparlectra's own scenario engine takes a branch out, started
from the base solution held in the nodes, the branch put back afterwards.
`rows`: 0-based branch rows of the `.m`; Sparlectra's importer makes
`branchVec[row + 1]` of each, asserted by its buses.
"""
struct Outages
    branches::Vector{Int}
    vm::Vector{Float64}   # base solution, written by solve_base!
    va::Vector{Float64}
end

function load_outages(m::Model, rows::AbstractVector, from::AbstractVector, to::AbstractVector)
    number(node) = Int(m.net.busOrigIdxDict[node])
    branches = [Int(r) + 1 for r in rows]
    for (b, f, t) in zip(branches, from, to)
        br = m.net.branchVec[b]
        @assert (number(br.fromBus), number(br.toBus)) == (Int(f), Int(t)) "branch row $(b - 1) is not $f-$t"
    end
    n = length(m.net.nodeVec)
    return Outages(branches, zeros(n), zeros(n))
end

function solve_base!(m::Model, o::Outages, tol::Float64, max_iterations::Int)
    steps = solve!(m, tol, max_iterations)
    o.vm .= [node._vm_pu for node in m.net.nodeVec]
    o.va .= [node._va_deg for node in m.net.nodeVec]
    return steps
end

function solve_outage!(m::Model, o::Outages, k::Int, tol::Float64, max_iterations::Int)
    br = m.net.branchVec[o.branches[k]]
    br.status, br.from_status, br.to_status = 0, 0, 0
    for (i, node) in enumerate(m.net.nodeVec)
        node._vm_pu = o.vm[i]
        node._va_deg = o.va[i]
        node._nodeType = m.types0[i]
    end
    steps = runpf!(m.net, tol, max_iterations, false)
    br.status, br.from_status, br.to_status = 1, 1, 1
    return steps
end

include("se.jl")

"""Versions of the packages that make up the tool, for the results' metadata."""
versions() = Dict(
    "Sparlectra" => pkgversion(Sparlectra),
    "AnalyticLoadFlow" => pkgversion(SP.AnalyticLoadFlow),
)

# Info and warning logs off: the importers log per case and per defect,
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
    demo = joinpath(pkgdir(Sparlectra), "data", "cgmes_demo", "sp_case14")
    profiles = [joinpath(demo, f) for f in readdir(demo) if endswith(f, ".xml") && !occursin("_SV", f)]
    @compile_workload begin
        Logging.with_logger(Logging.NullLogger()) do
            redirect_stdout(devnull) do
                m = load_matpower(path)
                solve!(m, 1e-8, 30)
                solution(m)
                sw = load_sweep(m, [2, 3], [0.0 0.0; 1.0 -1.0], [0.0 0.0; 0.5 -0.5], [2.0 -2.0; 0.0 0.0])
                apply!(m, sw, 2)
                solve!(m, 1e-8, 30)
                solution(m)
                o = load_outages(m, [0, 2], [1, 1], [2, 3])
                solve_base!(m, o, 1e-8, 30)
                solve_outage!(m, o, 1, 1e-8, 30)
                solution(m)
                # vm at 1, P/Q injections at 2 and 3, P/Q from-end flow on row 0
                se = load_se(path, [0, 1, 2, 1, 2, 3, 4], [1, 2, 2, 3, 3, 0, 0], [-1, -1, -1, -1, -1, 0, 0],
                             [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 2, 2],
                             [1.02, 50.0, 10.0, -90.0, -30.0, 20.0, 5.0], [0.004, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             1e-8, 30)
                estimate!(se)
                se_solution(se)
                m = load_cgmes(profiles)
                solve!(m, 1e-8, 30)
                solution(m)
            end
        end
    end
end

end
