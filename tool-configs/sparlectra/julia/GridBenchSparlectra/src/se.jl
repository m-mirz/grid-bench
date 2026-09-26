# State estimation: the Julia half of adapters/sparlectra_se_adapter.py
# (settings and their justification are documented there). Included by
# GridBenchSparlectra.jl, so it is compiled into the image with the rest.

# By code, in the order of adapters/sparlectra_se_adapter.py's KINDS.
const SE_TYPE = [SP.VmMeas, SP.PinjMeas, SP.QinjMeas, SP.PflowMeas, SP.QflowMeas]

struct SEModel
    pf::Model                          # the network as the power-flow half reads it
    meas::Vector{SP.Measurement}
    cfg::SP.StateEstimationConfig
    result::Base.RefValue{Any}
end

"""The `.m` as `load_matpower` reads it, plus one Sparlectra measurement per
row of the case's measurement set. Buses by MATPOWER number
(`busOrigIdxDict`); branches by `.m` row: the importer adds one branch per
row, in order (asserted), so row r (0-based) is `branchVec[r + 1]`, checked
against the measurement's from and to bus. `rows[i] < 0`: a bus measurement."""
function load_se(path::AbstractString, kinds, buses, rows, froms, tos, values, sigmas, tol::Float64,
                 max_iterations::Int)
    m = load_matpower(path)
    net = m.net
    node_of = Dict(orig => node for (node, orig) in net.busOrigIdxDict)
    meas = SP.Measurement[]
    for i in eachindex(kinds)
        typ = SE_TYPE[Int(kinds[i]) + 1]
        if rows[i] < 0
            SP.addMeasurement!(meas; typ = typ, value = values[i], sigma = sigmas[i], busIdx = node_of[Int(buses[i])])
        else
            k = Int(rows[i]) + 1
            br = net.branchVec[k]
            @assert (br.fromBus, br.toBus) == (node_of[Int(froms[i])], node_of[Int(tos[i])]) "branch row $(rows[i]) is not branchVec[$k]"
            SP.addMeasurement!(meas; typ = typ, value = values[i], sigma = sigmas[i], branchIdx = k, direction = :from)
        end
    end
    cfg = SP.StateEstimationConfig(;
        method = :wls,
        tol = tol,
        max_iter = max_iterations,
        flatstart = true,
        update_net = true,
        update_shunts = false,
        update_taps = false,
        robust = false,
        robust_mode = :off,
        max_eliminations = 0,
        topology_precheck = false,
    )
    return SEModel(m, meas, cfg, Ref{Any}(nothing))
end

"""Every node back to 1 p.u. and 0 degrees (the flat start reads the slack's
magnitude from its node, which the previous estimate overwrote), then one
estimate. Returns the iterations, or -1 if not converged."""
function estimate!(m::SEModel)
    for node in m.pf.net.nodeVec
        node._vm_pu = 1.0
        node._va_deg = 0.0
    end
    res = SP._runse_configured!(m.pf.net, m.meas, m.cfg)
    m.result[] = res
    return res.converged ? res.iterations : -1
end

"""MATPOWER bus numbers, |V| in p.u. and angle in degrees of the last estimate."""
function se_solution(m::SEModel)
    v = m.result[].voltages
    return m.pf.ids, [abs(v[k]) for k in m.pf.nodes], [rad2deg(angle(v[k])) for k in m.pf.nodes]
end
