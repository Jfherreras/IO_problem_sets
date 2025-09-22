
# ps1_code_full_v2.jl
# Complete PS solution with AD-safe Mixed Logit (MC & GH) signatures
using CSV, DataFrames, LinearAlgebra, Statistics, Random, Printf
using Optim, ForwardDiff, Distributions, FastGaussQuadrature
using StatsBase
using Plots

# ---------------------------
# 0. I/O and data preparation
# ---------------------------
if !isdefined(@__MODULE__, :DATA_PATH)
    const DATA_PATH = "schools_dataset.csv"
end
if !isdefined(@__MODULE__, :FIG_DIR)
    const FIG_DIR = "figures"
end
isdir(FIG_DIR) || mkdir(FIG_DIR)

df = CSV.read(DATA_PATH, DataFrame)
select!(df, Not(contains.(names(df), "Unnamed")))

N = length(unique(df.household_id))
J = length(unique(df.school_id))
@info "Loaded $(nrow(df)) rows ($N households × $J schools)."

sort!(df, [:household_id, :school_id])
x_test   = reshape(Matrix(select(df, :test_scores)), N, J)
x_sports = reshape(Matrix(select(df, :sports)), N, J)
dist     = reshape(Matrix(select(df, :distance)), N, J)
y        = reshape(Matrix(select(df, :y_ij)), N, J)
choice_ix = map(argmax, eachrow(y))

# ---------------------------
# Helpers
# ---------------------------
softmax_rows(V) = begin
    maxV = maximum(V, dims=2)
    expV = exp.(V .- maxV)
    denom = sum(expV, dims=2)
    expV ./ denom
end

# ---------------------------
# Q1. Distance histograms
# ---------------------------
function plot_q1!()
    isdir(FIG_DIR) || mkdir(FIG_DIR)
    all_d = vec(dist)
    chosen_d = [dist[i, choice_ix[i]] for i in 1:N]
    histogram(all_d, bins=40, xlabel="Distance (miles)", ylabel="Count",
        title="Distance to all schools", legend=false)
    savefig(joinpath(FIG_DIR, "q1_all_distances.png"))
    histogram(chosen_d, bins=40, xlabel="Distance (miles)", ylabel="Count",
        title="Distance to chosen school", legend=false)
    savefig(joinpath(FIG_DIR, "q1_chosen_distance.png"))
end

plot_q1!()

# ---------------------------
# Q2–Q4. Plain logit (NO ξ) — identified
# θ = [β1, β2, α]
# ---------------------------
function nll_plain_no_xi(θ::AbstractVector{T}) where {T<:Real}
    β1, β2, α = θ
    V = β1 .* x_test .+ β2 .* x_sports .- α .* dist
    P = softmax_rows(V)
    ϵ = T(1e-12)
    return -sum(y .* log.(P .+ ϵ))
end

function fit_plain_no_xi(; θ0 = zeros(3))
    opt = optimize(nll_plain_no_xi, θ0, BFGS(); autodiff=:forward)
    θhat = Optim.minimizer(opt)
    nll  = Optim.minimum(opt)
    H = ForwardDiff.hessian(nll_plain_no_xi, θhat) |> Symmetric
    vcov = try inv(H + 1e-8I) catch; pinv(H) end
    se = sqrt.(diag(vcov))
    β1, β2, α = θhat
    V = β1 .* x_test .+ β2 .* x_sports .- α .* dist
    P = softmax_rows(V)
    ŝ = vec(mean(P, dims=1))
    div12 = ŝ[2] / (1 - ŝ[1])
    elast = -α .* (1 .- [P[i, choice_ix[i]] for i in 1:N]) .* [dist[i, choice_ix[i]] for i in 1:N]
    avg_elast = mean(elast)
    return (name="Plain (no ξ)", θ=θhat, se=se, nll=nll, shares=ŝ, div12=div12, avg_own_elast=avg_elast)
end

res_plain = fit_plain_no_xi()

# ---------------------------
# Q5. Restricted model with ONLY ξ_j (no β1,β2,α)
# ---------------------------
function pack_xi(θ::AbstractVector{T}) where {T<:Real}
    ξ = zeros(T, J)
    ξ[1:J-1] .= θ
    return ξ
end

function nll_xi_only(θ::AbstractVector{T}) where {T<:Real}
    ξ = pack_xi(θ)
    V = ξ'
    P = softmax_rows(V)
    ϵ = T(1e-12)
    return -sum(y .* log.(P .+ ϵ))
end

function fit_xi_only()
    θ0 = zeros(J-1)
    opt = optimize(nll_xi_only, θ0, BFGS(); autodiff=:forward)
    θhat = Optim.minimizer(opt)
    nll  = Optim.minimum(opt)
    H = ForwardDiff.hessian(nll_xi_only, θhat) |> Symmetric
    vcov = try inv(H + 1e-8I) catch; pinv(H) end
    se = sqrt.(diag(vcov))
    ξ = pack_xi(θhat)
    P = softmax_rows(ξ')
    ŝ = vec(mean(P, dims=1))
    div12 = ŝ[2] / (1 - ŝ[1])
    return (name="Restricted (ξ only)", θ=θhat, se=se, nll=nll, shares=ŝ, div12=div12, avg_own_elast=NaN)
end

res_xionly = fit_xi_only()

# ---------------------------
# Q6–Q7. Mixed logit (random β1i ~ N(β1, σ_b)), fixed β2, α
# θ = [β1, logσ_b, β2, α] — log-parametrization keeps σ_b>0
# ---------------------------

# AD-safe: allow θ to be Dual, but draws/nodes can be Float64.
# We convert draws/nodes/weights to θ's element type T when needed.

function shares_mixed_given_draws(β1draws::AbstractMatrix{T}, β2::S, α::R) where {T<:Real,S<:Real,R<:Real}
    U = promote_type(T,S,R)
    β2u, αu = U(β2), U(α)
    Pavg = zeros(U, N, J)
    Rn = size(β1draws, 2)
    for r in 1:Rn
        β1r = β1draws[:, r]              # length N, eltype T
        V = (β1r .* x_test) .+ β2u .* x_sports .- αu .* dist
        Pavg .+= softmax_rows(V)
    end
    Pavg ./= U(Rn)
    return Pavg
end

function nll_msl_mc(θ::AbstractVector{T}, β1_draws::AbstractMatrix{S}) where {T<:Real,S<:Real}
    β1, logσb, β2, α = θ
    σb = exp(logσb)
    # convert draws to θ's type to match AD (T)
    β1draws_T = T.(β1_draws)             # N×R
    β1draws = β1 .+ σb .* β1draws_T
    Pavg = shares_mixed_given_draws(β1draws, β2, α)
    ϵ = T(1e-12)
    ll = sum(log.([Pavg[i, choice_ix[i]] + ϵ for i in 1:N]))
    return -ll
end

function nll_msl_ghq(θ::AbstractVector{T}, nodes::AbstractVector{S}, weights::AbstractVector{S}) where {T<:Real,S<:Real}
    β1, logσb, β2, α = θ
    σb = exp(logσb)
    nodesT = T.(nodes)
    weightsT = T.(weights)
    wnorm = weightsT ./ sqrt(T(pi))
    Pavg = zeros(T, N, J)
    for k in 1:length(nodesT)
        β1k = β1 .+ σb .* sqrt(T(2)) .* nodesT[k]
        V = (β1k .* x_test) .+ β2 .* x_sports .- α .* dist
        Pavg .+= wnorm[k] .* softmax_rows(V)
    end
    ϵ = T(1e-12)
    ll = sum(log.([Pavg[i, choice_ix[i]] + ϵ for i in 1:N]))
    return -ll
end

function fit_msl_mc(; R=100, seed=1, θ0 = [0.0, log(0.2), 0.0, 0.1])
    Random.seed!(seed)
    β1_draws = randn(N, R)               # standard normals (Float64)
    obj(θ) = nll_msl_mc(θ, β1_draws)
    opt = optimize(obj, θ0, BFGS(); autodiff=:forward)
    θhat = Optim.minimizer(opt)
    nll  = Optim.minimum(opt)
    H = ForwardDiff.hessian(obj, θhat) |> Symmetric
    vcov = try inv(H + 1e-8I) catch; pinv(H) end
    se = sqrt.(diag(vcov))
    # stats
    β1, logσb, β2, α = θhat
    σb = exp(logσb)
    Pavg = shares_mixed_given_draws((β1 .+ σb .* θhat[2]*0 .+ Float64.(β1_draws)) .* 0 .+ (β1 .+ σb .* Float64.(β1_draws)), β2, α)
    # The above line ensures Float64 broadcasting; simpler: recompute directly:
    Pavg = shares_mixed_given_draws((β1 .+ σb .* Float64.(β1_draws)), β2, α)
    ŝ = vec(mean(Pavg, dims=1))
    div12 = ŝ[2] / (1 - ŝ[1])
    elast = [-α * (1 - Pavg[i, choice_ix[i]]) * dist[i, choice_ix[i]] for i in 1:N]
    avg_elast = mean(elast)
    return (name="Mixed (MSL, MC R=$(R))", θ=θhat, se=se, nll=nll, shares=ŝ, div12=div12, avg_own_elast=avg_elast)
end

function fit_msl_ghq(; K=20, θ0 = [0.0, log(0.2), 0.0, 0.1])
    nodes, weights = gausshermite(K)     # Float64
    obj(θ) = nll_msl_ghq(θ, nodes, weights)
    opt = optimize(obj, θ0, BFGS(); autodiff=:forward)
    θhat = Optim.minimizer(opt)
    nll  = Optim.minimum(opt)
    H = ForwardDiff.hessian(obj, θhat) |> Symmetric
    vcov = try inv(H + 1e-8I) catch; pinv(H) end
    se = sqrt.(diag(vcov))
    # stats
    β1, logσb, β2, α = θhat
    σb = exp(logσb)
    nodesT = nodes; weightsT = weights ./ sqrt(pi)
    Pavg = zeros(Float64, N, J)
    for k in 1:length(nodesT)
        β1k = β1 .+ σb .* sqrt(2.0) .* nodesT[k]
        V = (β1k .* x_test) .+ β2 .* x_sports .- α .* dist
        Pavg .+= weightsT[k] .* softmax_rows(V)
    end
    ŝ = vec(mean(Pavg, dims=1))
    div12 = ŝ[2] / (1 - ŝ[1])
    elast = [-α * (1 - Pavg[i, choice_ix[i]]) * dist[i, choice_ix[i]] for i in 1:N]
    avg_elast = mean(elast)
    return (name="Mixed (MSL, Gauss–Hermite K=$(K))", θ=θhat, se=se, nll=nll, shares=ŝ, div12=div12, avg_own_elast=avg_elast)
end

res_msl_mc  = fit_msl_mc(R=100, seed=123)
res_msl_ghq = fit_msl_ghq(K=20)

# ---------------------------
# Q8–Q11. MSM estimator (same θ = [β1, logσ_b, β2, α])
# ---------------------------
function stacked_instruments()
    L = 4
    Z = Array{Float64,3}(undef, N, J, L)
    for i in 1:N, j in 1:J
        Z[i,j,1] = 1.0
        Z[i,j,2] = x_test[i,j]
        Z[i,j,3] = x_sports[i,j]
        Z[i,j,4] = dist[i,j]
    end
    return Z
end
const Z_ijl = stacked_instruments()
const L = size(Z_ijl, 3)

function Pbar_mixed(θ::AbstractVector{T}) where {T<:Real}
    β1, logσb, β2, α = θ
    σb = exp(logσb)
    nodes, weights = gausshermite(20)
    nodesT = T.(nodes)
    wnorm = T.(weights) ./ sqrt(T(pi))
    Pavg = zeros(T, N, J)
    for k in 1:length(nodesT)
        β1k = β1 .+ σb .* sqrt(T(2)) .* nodesT[k]
        V = (β1k .* x_test) .+ β2 .* x_sports .- α .* dist
        Pavg .+= wnorm[k] .* softmax_rows(V)
    end
    return Pavg
end

function moments_msm(θ::AbstractVector{T}) where {T<:Real}
    Pavg = Pbar_mixed(θ)
    m = zeros(T, J*L)
    idx = 1
    for j in 1:J, l in 1:L
        #m[idx] = mean(@view Z_ijl[:,j,l] .* (y[:,j] .- Pavg[:,j]))
        m[idx] = mean(Z_ijl[:,j,l] .* (y[:,j] .- Pavg[:,j]))
        idx += 1
    end
    return m
end

msm_objective(θ) = begin
    g = moments_msm(θ)
    dot(g, g)
end

function fit_msm(; θ0 = [res_msl_mc.θ...])
    # Stage 1: W = I
    opt1 = optimize(msm_objective, θ0, BFGS(); autodiff=:forward)
    θ1 = Optim.minimizer(opt1)
    # Build S at θ1
    P1 = Pbar_mixed(θ1)
    mlen = J*L
    M = zeros(mlen, N)
    for i in 1:N
        idx = 1
        for j in 1:J, l in 1:L
            M[idx, i] = Z_ijl[i,j,l] * (y[i,j] - P1[i,j])
            idx += 1
        end
    end
    S = cov(Matrix(M)'; corrected=false)
    Winv = try inv(S + 1e-8I) catch; pinv(S) end
    # Stage 2: efficient W
    msm_objective2(θ) = begin
        g = moments_msm(θ)
        g' * Winv * g
    end
    opt2 = optimize(msm_objective2, θ1, BFGS(); autodiff=:forward)
    θ2 = Optim.minimizer(opt2)
    # Asy. variance
    Jg2 = ForwardDiff.jacobian(moments_msm, θ2)
    B = Jg2' * Winv * Jg2
    A = Jg2' * Winv * S * Winv * Jg2
    vcov = try inv(B + 1e-8I) * A * inv(B + 1e-8I) catch; pinv(B) * A * pinv(B) end
    se = sqrt.(diag(vcov))
    # Reporting
    P2 = Pbar_mixed(θ2)
    ŝ = vec(mean(P2, dims=1))
    div12 = ŝ[2] / (1 - ŝ[1])
    β1, logσb, β2, α = θ2
    elast = [-α * (1 - P2[i, choice_ix[i]]) * dist[i, choice_ix[i]] for i in 1:N]
    avg_elast = mean(elast)
    nll_proxy = -sum(log.([P2[i, choice_ix[i]] + 1e-12 for i in 1:N]))
    return (name="MSM (efficient 2-step)", θ=θ2, se=se, nll=nll_proxy, shares=ŝ, div12=div12, avg_own_elast=avg_elast),
           (name="MSM (first stage, W=I)", θ=θ1, se=nothing, nll=nll_proxy, shares=ŝ, div12=div12, avg_own_elast=avg_elast)
end

res_msm2, res_msm1 = fit_msm()

# ---------------------------
# Print results
# ---------------------------
function print_fit(res)
    println("=== ", res.name, " ===")
    @printf("nll: %.3f\n", res.nll)
    println("θ̂: ", res.θ')
    if res.se !== nothing; println("se: ", res.se'); end
    println("shares: ", res.shares')
    @printf("diversion 1→2: %.4f\n", res.div12)
    @printf("avg own elasticity (chosen, distance): %.4f\n\n", res.avg_own_elast)
end

print_fit(res_plain)
print_fit(res_xionly)
print_fit(res_msl_mc)
print_fit(res_msl_ghq)
print_fit(res_msm1)
print_fit(res_msm2)

using CSV
table = DataFrame(model=String[], nll=Float64[], div12=Float64[], avg_own_elast=Float64[])
for r in (res_plain, res_xionly, res_msl_mc, res_msl_ghq, res_msm2)
    push!(table, (r.name, r.nll, r.div12, r.avg_own_elast))
end
CSV.write("results_table.csv", table)
@info "Wrote results_table.csv and Q1 figures. Done."
