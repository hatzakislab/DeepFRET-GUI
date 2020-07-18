from PyInstaller.utils.hooks import collect_submodules

# hiddenimports = [
#     'pomegranate.utils',
#     'pomegranate.base'
#     'pomegranate.bayes',
#     'pomegranate.distributions.NeuralNetworkWrapper',
#     'pomegranate.distributions.distributions',
#     'pomegranate.distributions.UniformDistribution',
#     'pomegranate.distributions.BernoulliDistribution',
#     'pomegranate.distributions.BetaDistribution',
#     'pomegranate.distributions.ConditionalProbabilityTable',
#     'pomegranate.distributions.DirichletDistribution',
#     'pomegranate.distributions.DiscreteDistribution',
#     'pomegranate.distributions.distributions',
#     'pomegranate.distributions.ExponentialDistribution',
#     'pomegranate.distributions.GammaDistribution',
#     'pomegranate.distributions.IndependentComponentsDistribution',
#     'pomegranate.distributions.JointProbabilityTable',
#     'pomegranate.distributions.KernelDensities',
#     'pomegranate.distributions.LogNormalDistribution',
#     'pomegranate.distributions.MultivariateGaussianDistribution',
#     'pomegranate.distributions.NormalDistribution',
#     'pomegranate.distributions.PoissonDistribution',
# ]

hiddenimports = collect_submodules('pomegranate')
hiddenimports += collect_submodules('pomegranate.utils')
hiddenimports += collect_submodules('pomegranate.distributions')
hiddenimports += collect_submodules('networkx')