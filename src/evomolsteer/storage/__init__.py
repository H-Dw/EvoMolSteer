"""Lossless storage; online generation does not import the feature-table stack."""
from importlib import import_module

_EXPORTS={'TrajectoryPackage':'trajectory','pack_trajectory':'trajectory','pack_arrays':'trajectory',
          'FeaturePackage':'features','pack_features':'features','StepTrajectoryWriter':'streaming',
          'convert_trajectory':'lifecycle','convert_dataset':'lifecycle'}
__all__=list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    return getattr(import_module('.'+_EXPORTS[name],__name__),name)
