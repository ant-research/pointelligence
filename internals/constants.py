import torch


class Constants(object):
    __2d_shifts = dict()
    __3d_shifts = dict()
    __zero = dict()
    __one = dict()
    __range = dict()

    @classmethod
    def get_2d_shifts(cls, device):
        if device not in cls.__2d_shifts:
            cls.__2d_shifts[device] = torch.tensor(
                [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int32, device=device
            )
        return cls.__2d_shifts[device]

    @classmethod
    def get_3d_shifts(cls, device):
        if device not in cls.__3d_shifts:
            cls.__3d_shifts[device] = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.int32,
                device=device,
            )
        return cls.__3d_shifts[device]

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

    @classmethod
    def get_range(cls, device, dtype=torch.int64):
        k = (device, dtype)
        if k not in cls.__range:
            range_max = 1048576  # 2^20
            cls.__range[k] = torch.arange(range_max, dtype=dtype, device=device)
        return cls.__range[k]
