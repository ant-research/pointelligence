try:
    from .msc_v1m2_pointcnnpp import MaskedSceneContrast as MSC_v1m2_pointcnnpp
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import msc_v1m2_pointcnnpp: {e}", ImportWarning)
    MSC_v1m2_pointcnnpp = None




