module DMM
    using Distributions
    using ConjugatePriors

    import Distributions:
        Distribution,
        UnivariateDistribution,
        MultivariateDistribution,
        NormalCanon,
        Normal,
        Gamma,
        Exponential,
        MvNormal,
        MvNormalCanon,
        GenericMvTDist,
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
    include("./DMMState.jl")
    include("./DPCluster.jl")
end
