module DMM
    using Distributions
    using ConjugatePriors

    import Distributions:
        Distribution,
        UnivariateDistribution,
        MultivariateDistribution,
        NormalCanon,
        Normal,
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

    include("./Models/uv_normal.jl")
    include("./Models/mv_normal.jl")
    include("./DMMState.jl")
    include("./DPCluster.jl")
end
