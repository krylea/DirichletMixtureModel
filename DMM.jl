module DMM
    using Distributions
    using ConjugatePriors
    using PDMats

    import PDMats: PDMat

    import Distributions:
        Distribution,
        UnivariateDistribution,
        MultivariateDistribution,
        NormalCanon,
        Normal,
        NormalKnownSigma,
        Gamma,
        Exponential,
        MvNormal,
        MvNormalCanon,
        GenericMvTDist,
        MvNormalStats,
        logmvgamma,
        suffstats,
        pdf

    import ConjugatePriors:
        NormalGamma,
        NormalWishart,
        rand,
        pdf,
        logpdf,
        posterior_canon

    include("./models/uv_normal.jl")
    include("./models/mv_normal.jl")
    include("./models/uv_exp.jl")
    include("./model.jl")
    include("./DMMState.jl")
    include("./DPCluster.jl")
end
