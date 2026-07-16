import torch


class Constants(object):
    __3d_offset_cubes = dict()
    __zero = dict()
    __one = dict()

    @classmethod
    def get_3d_offset_cube(cls, device, lower, upper):
        """Cache the small int64 Cartesian offset cube used by grid search."""
        key = (device, int(lower), int(upper))
        if key not in cls.__3d_offset_cubes:
            values = torch.arange(lower, upper + 1, dtype=torch.int64, device=device)
            x, y, z = torch.meshgrid(values, values, values, indexing="ij")
            cls.__3d_offset_cubes[key] = torch.stack(
                (x.reshape(-1), y.reshape(-1), z.reshape(-1)), dim=1
            )
        return cls.__3d_offset_cubes[key]

    @classmethod
    def get_zero(cls, device, dtype):
        k = (device, dtype)
        if k not in cls.__zero:
            cls.__zero[k] = torch.zeros((1,), dtype=dtype, device=device)
        return cls.__zero[k]

    @classmethod
    def get_one(cls, device, dtype):
        k = (device, dtype)
        if k not in cls.__one:
            cls.__one[k] = torch.ones((1,), dtype=dtype, device=device)
        return cls.__one[k]
