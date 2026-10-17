"""
The Julia half of adapters/examodelspower_opf_adapter.py (settings and their
justification are documented there). It lives in the image, not in
adapters/, because it is compiled into it: the workload at the bottom runs
load, solve and solution on a 3-bus case at build time, so every fresh
process starts with native code instead of spending its first case in the
JIT. Changing this file needs `docker/build.sh examodelspower`.

It depends on PythonCall (unused here) for the same reason as
GridBenchSienna: juliacall loads PythonCall first, and code compiled without
it is invalidated by PythonCall's method definitions.
"""
module GridBenchExaModelsPower

using Logging
using PrecompileTools: @compile_workload, @setup_workload
using PythonCall: PythonCall
import ExaModels
import ExaModelsPower as EMP
import MadNLP

struct Model
    nlp::ExaModels.ExaModel
    vars::NamedTuple
    data::NamedTuple
    tol::Float64
    max_iterations::Int
    stats::Base.RefValue{Any}
end

"""Parses the `.m` (ExaPowerIO, ExaModelsPower's parser) and builds the
polar AC-OPF once, as `ac_opf_model(path)` does (the same two calls, kept
apart so the parsed data stays at hand for the joins). Then writes the
common flat start into the model's starting point: every bus at 1 p.u. and
0 rad (ExaModelsPower's own start), every generator at the middle of its P
and Q ranges (its own start is 0)."""
function load(path::AbstractString, tol::Float64, max_iterations::Int)
    data = EMP.parse_ac_power_data(String(path))
    nlp, vars, _ = EMP.build_polar_opf(data, EMP.dummy_extension)
    x0 = nlp.meta.x0
    ng = length(data.gen)
    x0[vars.pg.offset .+ (1:ng)] .= (data.pmin .+ data.pmax) ./ 2
    x0[vars.qg.offset .+ (1:ng)] .= (data.qmin .+ data.qmax) ./ 2
    return Model(nlp, vars, data, tol, max_iterations, Ref{Any}(nothing))
end

"""One solve from the start set in `load`: a new MadNLP solver on the
persistent model each time (`madnlp`, as the package is used), so no
iterate, dual or barrier parameter carries over from the previous solve.
Returns MadNLP's iteration count and its status, by name."""
function solve!(m::Model)
    stats = MadNLP.madnlp(m.nlp; linear_solver = MadNLP.MumpsSolver, tol = m.tol, max_iter = m.max_iterations,
                          print_level = MadNLP.ERROR)
    m.stats[] = stats
    return stats.iter, string(stats.status)
end

"""Bus numbers, |V| in p.u., angles in degrees; then generator positions
(0-based, among the in-service generators in the `.m`'s order: ExaPowerIO
drops the others and numbers the rest), their bus numbers, P in MW and Q in
MVAr, of the last solve."""
function solution(m::Model)
    stats, v, d = m.stats[], m.vars, m.data
    base_mva = d.baseMVA[1]
    return ([string(b.bus_i) for b in d.bus],
            ExaModels.solution(stats, v.vm),
            rad2deg.(ExaModels.solution(stats, v.va)),
            [g.i - 1 for g in d.gen],
            [d.bus[g.bus].bus_i for g in d.gen],
            ExaModels.solution(stats, v.pg) .* base_mva,
            ExaModels.solution(stats, v.qg) .* base_mva)
end

"""Versions of the packages that make up the tool, for the results' metadata."""
versions() = Dict(
    "ExaModelsPower" => pkgversion(EMP),
    "ExaModels" => pkgversion(ExaModels),
    "ExaPowerIO" => pkgversion(EMP.ExaPowerIO),
    "MadNLP" => pkgversion(MadNLP),
)

# ExaPowerIO logs "Loading matpower file" at Info on every parse; the parse is timed.
quiet() = Logging.disable_logging(Logging.Info)

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
        Logging.with_logger(Logging.NullLogger()) do
            m = load(path, 1e-6, 200)
            solve!(m)
            solution(m)
        end
    end
end

end
