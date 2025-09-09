###########################
# Empirical IO — PS0 (Numerical)
# Julia solutions (ASCII-only, reproducible)
#
# Requires (once):
#   ] add Distributions FastGaussQuadrature QuadGK
###########################

using LinearAlgebra
using Random
using Distributions
using FastGaussQuadrature
using QuadGK
using Printf

# -------------------------
# Global reproducibility seed
# -------------------------
const SEED = 123456
Random.seed!(SEED)

# -------------------------
# Optional: StatsFuns for logsumexp comparison in Part 0, Ex.3
# (If not installed, the code will skip that comparison gracefully.)
# -------------------------
const HAS_STATSFUNS = let path = Base.find_package("StatsFuns")
    if path === nothing
        false
    else
        # load at top-level (safe), not inside a function
        @eval using StatsFuns
        true
    end
end

# -------------------------
# Utilities
# -------------------------
# Numerically stable logistic (works for scalars and arrays)
logistic(z::Real) = z >= 0 ? 1/(1+exp(-z)) : exp(z)/(1+exp(z))
logistic(z::AbstractArray) = logistic.(z)

# Softmax (stable) used in Part 0 Ex.1 checks
function softmax(x::AbstractVector)
    m = maximum(x)
    ex = exp.(x .- m)
    ex ./ sum(ex)
end

# log-sum-exp, naive and max-trick
logsumexp_naive(x::AbstractVector) = log(sum(exp.(x)))
function logsumexp_maxtrick(x::AbstractVector)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

# Pretty printing helpers (loose types to avoid MethodError)
function print_kv(title, pairs)
    println(title)
    for (k,v) in pairs
        @printf("  %-28s : %s\n", k, string(v))
    end
    println()
end

function print_table(title, header, rows)
    println(title)
    @printf("%-28s  %-10s  %-18s  %-18s\n", header...)
    println("-"^82)
    for r in rows
        @printf("%-28s  %-10s  %-18.12f  %-18.12f\n", r[1], r[2], r[3], r[4])
    end
    println()
end

# =========================================================
# Part 0 — Logit Inclusive Value
# =========================================================

# -------------------------
# Part 0 — Exercise 1:
# Convexity of IV = log(sum_i exp(x_i)) (with x_0 = 0)
# -------------------------
# Sketch (comments): gradient p = softmax(x), Hessian H = diag(p) - p*p'
# For any v, v' H v = sum_i p_i v_i^2 - (sum_i p_i v_i)^2 = Var_p(v) >= 0.
# We also provide a small numerical PSD check.

function part0_ex1()
    println("Part 0 — Exercise 1: Convexity of log-sum-exp (numerical PSD checks)")
    samples = [[0.0, 1.0, 2.0],
               [-5.0, 0.0, 3.5, 6.0],
               [0.0, 10.0, 600.0]]
    for (k, x) in enumerate(samples)
        p = softmax(x)
        H = Diagonal(p) - p * transpose(p)
        eigs = eigvals(Symmetric(H))
        @printf("  Sample %d: min eigenvalue of Hessian = %.6e  (>= 0 up to numerical tol)\n",
                k, minimum(eigs))
    end
    println()
end

# -------------------------
# Part 0 — Exercise 2:
# Implement the max-trick for IV
# -------------------------
function part0_ex2()
    println("Part 0 — Exercise 2: Max-trick implementation and examples")
    examples = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 10.0, 600.0],
        [0.0, -1000.0, 1.0],
    ]
    for (i, x) in enumerate(examples)
        naive = try
            logsumexp_naive(x)
        catch e
            e
        end
        mt = logsumexp_maxtrick(x)
        @printf("  Example %d: x = %s\n", i, string(x))
        @printf("    naive logsumexp        = %s\n", string(naive))
        @printf("    max-trick logsumexp    = %.12f\n\n", mt)
    end
end

# -------------------------
# Part 0 — Exercise 3:
# Compare to a library implementation (StatsFuns.logsumexp) if available
# -------------------------
function part0_ex3()
    println("Part 0 — Exercise 3: Compare to library logsumexp (if available)")
    x = [0.0, 10.0, 600.0]
    ours = logsumexp_maxtrick(x)
    if HAS_STATSFUNS
        theirs = StatsFuns.logsumexp(x)
        @printf("  ours (max-trick)    = %.12f\n", ours)
        @printf("  StatsFuns.logsumexp = %.12f\n", theirs)
        println("  Match? ", isapprox(ours, theirs; rtol=1e-12, atol=1e-12))
    else
        @printf("  ours (max-trick)    = %.12f\n", ours)
        println("  StatsFuns not installed; skipping direct comparison.")
    end
    println()
end

# =========================================================
# Part 1 — Markov Chains
# =========================================================

# Compute ergodic distribution via eigenvector; compare to rows of P^100
function stationary_from_eigs(P::AbstractMatrix)
    F = eigen(transpose(P))
    vals, vecs = F.values, F.vectors
    idx = argmin(abs.(vals .- 1))
    v = real.(vecs[:, idx])
    if any(v .< 0)
        v = -v
    end
    v ./ sum(v)
end

function part1_all()
    println("Part 1: Markov chains — ergodic distribution vs P^100")
    P = [0.2 0.4 0.4;
         0.1 0.3 0.6;
         0.5 0.1 0.4]
    pi_vec = stationary_from_eigs(P)
    P100 = P^100
    print_kv("  Results", [
        ("pi", pi_vec),
        ("row 1 of P^100", vec(P100[1, :])),
        ("row 2 of P^100", vec(P100[2, :])),
        ("row 3 of P^100", vec(P100[3, :]))
    ])
end

# =========================================================
# Part 2 — Numerical Integration
# =========================================================

# (PDF constants) 1-D: mu=0.5, sigma=2.0, X=0.5
const mu1d = 0.5
const sigma1d = 2.0
const X1d = 0.5
const dist1d = Normal(mu1d, sigma1d)

# Exercise 1: Create binomiallogit(beta; pdf) (the integrand)
binomiallogit(beta; pdf = (b -> pdf(dist1d, b))) = logistic(beta * X1d) * pdf(beta)

# Exercise 2: Integrate with quadgk (treat as "true")
function part2_ex2_true()
    val, _ = quadgk(b -> binomiallogit(b), -Inf, Inf; rtol=1e-14, atol=1e-14, maxevals=10^7)
    return val
end

# Exercise 3: Monte Carlo with 20 and 400 draws (reproducible via SEED)
function part2_ex3_mc(n::Int; seed::Int=SEED)
    rng = MersenneTwister(seed)
    draws = rand(rng, dist1d, n)
    mean(logistic.(draws .* X1d))
end

# Exercise 4–5: Gauss–Hermite for k = 4, 5, 11, 12 and weight-sum check
function gh_1d(n::Int)
    # Nodes/weights for ∫ e^{-x^2} f(x) dx ≈ Σ w_i f(x_i)
    x, w = gausshermite(n)
    beta = mu1d .+ sqrt(2.0) * sigma1d .* x
    est = (1.0 / sqrt(pi)) * sum(w .* logistic.(beta .* X1d))
    weight_sum = (1.0 / sqrt(pi)) * sum(w)   # should be 1 for f(x) = 1
    return est, weight_sum
end

# Exercise 6: Repeat in 2D with mu=(0.5,1), sigma=(2,1), X=(0.5,1)
const mu2d = (0.5, 1.0)
const sigma2d = (2.0, 1.0)
const X2d = (0.5, 1.0)
const dist2d_b1 = Normal(mu2d[1], sigma2d[1])
const dist2d_b2 = Normal(mu2d[2], sigma2d[2])

# MC in 2D
function mc_2d(n::Int; seed::Int=SEED)
    rng = MersenneTwister(seed)
    b1 = rand(rng, dist2d_b1, n)
    b2 = rand(rng, dist2d_b2, n)
    mean(logistic.(b1 .* X2d[1] .+ b2 .* X2d[2]))
end

# GH in 2D (tensor product); also returns (1/pi)*sum(w1)*sum(w2) check == 1
function gh_2d(n1::Int, n2::Int)
    x1, w1 = gausshermite(n1)
    x2, w2 = gausshermite(n2)
    beta1 = mu2d[1] .+ sqrt(2.0) * sigma2d[1] .* x1
    beta2 = mu2d[2] .+ sqrt(2.0) * sigma2d[2] .* x2
    B1 = reshape(beta1 .* X2d[1], :, 1)
    B2 = reshape(beta2 .* X2d[2], 1, :)
    G  = logistic.(B1 .+ B2)
    W  = w1 * transpose(w2)
    est = (1.0 / pi) * sum(W .* G)
    weight_sum = (1.0 / pi) * sum(w1) * sum(w2)  # should be 1 for f(x,y)=1
    return est, weight_sum
end

# Exercise 7: Tables for 1D and 2D (errors vs "true")
function part2_ex7_tables()
    # 1-D
    true1d = part2_ex2_true()
    mc20  = part2_ex3_mc(20;  seed=SEED)
    mc400 = part2_ex3_mc(400; seed=SEED)
    gh4,  wsum4  = gh_1d(4)
    gh5,  wsum5  = gh_1d(5)
    gh11, wsum11 = gh_1d(11)
    gh12, wsum12 = gh_1d(12)

    rows1 = [
        ["True (quadgk tol=1e-14)", "adaptive", true1d, 0.0],
        ["Monte Carlo (n=20)",      "20",       mc20,   abs(mc20 - true1d)],
        ["Monte Carlo (n=400)",     "400",      mc400,  abs(mc400 - true1d)],
        ["Gauss-Hermite (n=4)",     "4",        gh4,    abs(gh4 - true1d)],
        ["Gauss-Hermite (n=5)",     "5",        gh5,    abs(gh5 - true1d)],
        ["Gauss-Hermite (n=11)",    "11",       gh11,   abs(gh11 - true1d)],
        ["Gauss-Hermite (n=12)",    "12",       gh12,   abs(gh12 - true1d)],
    ]
    print_table("Part 2 — Exercise 7 (1-D): Results and absolute errors",
                ["method", "points", "estimate", "abs_error"], rows1)
    @printf("  GH weight-sum checks (should be 1): n=4 -> %.12f, n=5 -> %.12f, n=11 -> %.12f, n=12 -> %.12f\n\n",
            wsum4, wsum5, wsum11, wsum12)

    # 2-D: use high-order GH as 'true' to keep runtime reasonable
    true2d, wsum_true = gh_2d(35, 35)
    gh4x4,  wsum4x4  = gh_2d(4, 4)
    gh5x5,  wsum5x5  = gh_2d(5, 5)
    gh11x11,wsum11x11= gh_2d(11, 11)
    gh12x12,wsum12x12= gh_2d(12, 12)
    mc20_2d  = mc_2d(20;  seed=SEED)
    mc400_2d = mc_2d(400; seed=SEED)

    rows2 = [
        ["True ≈ GH (35x35)", "1225",  true2d, 0.0],
        ["Monte Carlo (n=20)", "20",    mc20_2d,   abs(mc20_2d - true2d)],
        ["Monte Carlo (n=400)","400",   mc400_2d,  abs(mc400_2d - true2d)],
        ["Gauss-Hermite (4x4)","16",    gh4x4,     abs(gh4x4 - true2d)],
        ["Gauss-Hermite (5x5)","25",    gh5x5,     abs(gh5x5 - true2d)],
        ["Gauss-Hermite (11x11)","121", gh11x11,   abs(gh11x11 - true2d)],
        ["Gauss-Hermite (12x12)","144", gh12x12,   abs(gh12x12 - true2d)],
    ]
    print_table("Part 2 — Exercise 7 (2-D): Results and absolute errors",
                ["method", "points", "estimate", "abs_error"], rows2)
    @printf("  GH weight-sum checks (should be 1): 4x4 -> %.12f, 5x5 -> %.12f, 11x11 -> %.12f, 12x12 -> %.12f; 35x35 (true) -> %.12f\n\n",
            wsum4x4, wsum5x5, wsum11x11, wsum12x12, wsum_true)
end

# ------- Write Part 2, Exercise 7 LaTeX tables -------
function write_part2_ex7_tables_tex(path::AbstractString = "part2_ex7_tables.tex")
    # 1D "true" + methods
    true1d = part2_ex2_true()
    mc20  = part2_ex3_mc(20;  seed=SEED)
    mc400 = part2_ex3_mc(400; seed=SEED)
    gh4,  _  = gh_1d(4)
    gh5,  _  = gh_1d(5)
    gh11, _  = gh_1d(11)
    gh12, _  = gh_1d(12)

    rows1 = [
        ("True (quadgk tol=1e-14)", "adaptive", true1d, 0.0),
        ("Monte Carlo (n=20)",      "20",       mc20,   abs(mc20 - true1d)),
        ("Monte Carlo (n=400)",     "400",      mc400,  abs(mc400 - true1d)),
        ("Gauss--Hermite (n=4)",    "4",        gh4,    abs(gh4 - true1d)),
        ("Gauss--Hermite (n=5)",    "5",        gh5,    abs(gh5 - true1d)),
        ("Gauss--Hermite (n=11)",   "11",       gh11,   abs(gh11 - true1d)),
        ("Gauss--Hermite (n=12)",   "12",       gh12,   abs(gh12 - true1d)),
    ]

    # 2D "true" via high-order GH + methods (independent normals)
    true2d, _ = gh_2d(35, 35)
    mc20_2d   = mc_2d(20;  seed=SEED)
    mc400_2d  = mc_2d(400; seed=SEED)
    gh4x4,  _ = gh_2d(4, 4)
    gh5x5,  _ = gh_2d(5, 5)
    gh11x11,_ = gh_2d(11, 11)
    gh12x12,_ = gh_2d(12, 12)

    rows2 = [
        ("True \\(\\approx\\) GH (35x35)", "1225", true2d, 0.0),
        ("Monte Carlo (n=20)",  "20",   mc20_2d,   abs(mc20_2d - true2d)),
        ("Monte Carlo (n=400)", "400",  mc400_2d,  abs(mc400_2d - true2d)),
        ("Gauss--Hermite (4x4)","16",   gh4x4,     abs(gh4x4 - true2d)),
        ("Gauss--Hermite (5x5)","25",   gh5x5,     abs(gh5x5 - true2d)),
        ("Gauss--Hermite (11x11)","121",gh11x11,   abs(gh11x11 - true2d)),
        ("Gauss--Hermite (12x12)","144",gh12x12,   abs(gh12x12 - true2d)),
    ]

    fmt(x) = @sprintf("%.6f", x)

    header = """
\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{booktabs}
\\usepackage{siunitx}
\\sisetup{round-mode=places, round-precision=6, table-number-alignment=center}
\\title{PS0 -- Part 2, Exercise 7 Tables}
\\date{}
\\begin{document}
\\maketitle
"""

    table1 = """
\\section*{1D Integration (Logit--Normal)}
\\begin{table}[h]
\\centering
\\begin{tabular}{@{}l c S[table-format=1.6] S[table-format=1.6]@{}}
\\toprule
Method & Points & {Estimate} & {Abs.\\ Error} \\\\
\\midrule
$(join(["$(r[1]) & $(r[2]) & $(fmt(r[3])) & $(fmt(r[4])) \\\\"
        for r in rows1], "\\n"))
\\bottomrule
\\end{tabular}
\\end{table}
"""

    table2 = """
\\section*{2D Integration (Logit--Normal, independent normals)}
\\begin{table}[h]
\\centering
\\begin{tabular}{@{}l c S[table-format=1.6] S[table-format=1.6]@{}}
\\toprule
Method & Points & {Estimate} & {Abs.\\ Error} \\\\
\\midrule
$(join(["$(r[1]) & $(r[2]) & $(fmt(r[3])) & $(fmt(r[4])) \\\\"
        for r in rows2], "\\n"))
\\bottomrule
\\end{tabular}
\\end{table}
"""

    footer = """
\\end{document}
"""

    open(path, "w") do io
        write(io, header)
        write(io, table1)
        write(io, table2)
        write(io, footer)
    end
    println("Wrote LaTeX tables to: ", path)
end



# Exercise 8: Vectorized mixture function (no loops)
function binomiallogitmixture(Xvec::AbstractVector; mu::Real=mu1d, sigma::Real=sigma1d, n::Int=12)
    x, w = gausshermite(n)
    beta = mu .+ sqrt(2.0) * sigma .* x
    Z = beta * transpose(Xvec)      # (n, m)
    G = logistic.(Z)
    vec((transpose(w) * G) ./ sqrt(pi))
end

function part2_ex8_demo()
    println("Part 2 — Exercise 8: binomiallogitmixture (vectorized; no loops)")
    X_demo = [0.1, 0.5, 1.0, 2.0]
    mix = binomiallogitmixture(X_demo; n=12)
    print_kv("  Demo output",
             [("X = " * string(X_demo), mix)])
end

# -------------------------
# Main driver (prints each exercise in order)
# -------------------------
function main()
    # Part 0
    part0_ex1()
    part0_ex2()
    part0_ex3()

    # Part 1
    part1_all()

    # Part 2
    println("Part 2 — Exercise 1: Define binomiallogit (done above as a function).")
    println()
    # Ex 2 (true), Ex 3 (MC), Ex 4–5 (GH + weight checks), Ex 6 (2D), Ex 7 (tables), Ex 8 (mixture)
    part2_ex7_tables()
    part2_ex8_demo()
end

# Execute if file is run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
