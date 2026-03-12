import torch

from .constants import Constants


def compute_overlapped_grid_indices(points, grid_size, overlap, dtype=torch.int32):
    assert 0 <= overlap < grid_size

    grid_size -= overlap
    if overlap > 0:
        channels = points.shape[-1]
        assert 2 == channels or 3 == channels
        shifts = (
            Constants.get_3d_shifts(points.device)
            if 3 == channels
            else Constants.get_2d_shifts(points.device)
        )
        shifts = (shifts - 0.5).mul_(overlap)
        points = (points.unsqueeze(dim=1) + shifts).reshape((-1, channels))

        grid_inds = points.div_(grid_size, rounding_mode="floor")
    else:
        grid_inds = points.div(grid_size, rounding_mode="floor")

    return grid_inds.contiguous().to(dtype)


def compute_grid_indices(
    points, grid_size, sample_inds=None, with_shifts=False, dtype=torch.int32
):
    """make points to 3d grid indices"""
    device = points.device

    if with_shifts:
        shifts = Constants.get_3d_shifts(device)
        num_shifts = 8

        # shift points half grid_size and compute 3d grid_inds [pts_num, 3]
        grid_inds = (
            (points - grid_size / 2).div_(grid_size, rounding_mode="floor").to(dtype)
        )
        # get pts and pts shift 3d grid_inds [pts_num, 8, 3] -> [pts_num * 8, 3]
        grid_inds = torch.reshape(grid_inds.unsqueeze(1) + shifts.unsqueeze(0), (-1, 3))

        if sample_inds is not None:
            # repeat each sample ind num_shifts times
            sample_inds = torch.reshape(
                sample_inds.unsqueeze(-1).expand(-1, num_shifts), (-1,)
            )
    else:
        # compute 3d grid_inds [pts_num, 3]
        grid_inds = points.div(grid_size, rounding_mode="floor").to(dtype)

    if sample_inds is not None:
        grid_inds = torch.cat((grid_inds, sample_inds.unsqueeze(-1)), dim=1)

    return grid_inds.contiguous()


def reduce_indices_to_1d(inds, inds_min=None, inds_stride=None, dtype=None):
    assert (
        inds.dtype is not torch.float16
    ), "reduce_inds_to_1d does not support half inds, because of the inds_stride[-1] > 32767 make inf!"
    inds_min = torch.amin(inds, dim=0, keepdim=True) if inds_min is None else inds_min
    if inds_stride is None:
        inds_step = torch.amax(inds, dim=0, keepdim=True).add_(1)
        if not (isinstance(inds_min, int) and 0 == inds_min):
            inds_step = inds_step.sub_(inds_min)
        inds_stride = inds_step.cumprod(dim=-1, dtype=torch.int64)
        assert inds_step.cumprod(dim=-1, dtype=torch.float64)[0, -1] < 2**63 - 1
        dtype = torch.int64 if inds_stride[0, -1] > 2**31 - 1 else torch.int32
        inds_stride = inds_stride.to(dtype)
        inds_stride = inds_stride.roll(1, dims=-1)
        inds_stride[0, 0] = 1

    if not (isinstance(inds_min, int) and 0 == inds_min):
        inds_ = (inds - inds_min).to(dtype).mul_(inds_stride)
    else:
        if dtype == inds.dtype:  # do not use in-place mul here!
            inds_ = inds.mul(inds_stride)
        else:  # to(dtype) creates a tensor with new storage, thus it is safe to use in-place mul_ here.
            inds_ = inds.to(dtype).mul_(inds_stride)
    inds_1d = torch.sum(inds_, dim=-1)
    return inds_1d, inds_min, inds_stride, dtype


class LookupStruct(object):
    def __init__(self, grid_inds_sorted_unique, grid_inds_min, grid_inds_stride, dtype):
        self.grid_inds_sorted_unique = grid_inds_sorted_unique
        self.grid_inds_min = grid_inds_min
        self.grid_inds_stride = grid_inds_stride
        self.dtype = dtype

    def size(self):
        return self.grid_inds_sorted_unique.numel()


def build_lookup_struct(
    grid_inds, return_unique_inds=False, return_sorter=False, return_unique_counts=False
):
    grid_inds_1d, grid_inds_min, grid_inds_stride, dtype = reduce_indices_to_1d(
        grid_inds
    )

    # range:                    0 1 2 3 4 5 6 7 8
    # grid_inds_1d:             3 1 1 9 7 4 1 3 7
    # grid_inds_sorted:         1 1 1 3 3 4 7 7 9
    # grid_inds_sorter:         1 2 6 0 7 5 4 8 3
    # grid_inds_sorted_unique:  - - 1 - 3 4 - 7 9
    # lookup_inds_sorted:       0 0 0 1 1 2 3 3 4
    # lookup_inds:              1 0 0 4 3 2 0 1 3
    # counts:                   - - 3 - 2 1 - 2 1
    # unique_inds_sorted:       - - 2 - 4 5 - 7 8
    # unique_inds:              - - 6 - 7 5 - 8 3
    grid_inds_sorted, grid_inds_sorter = torch.sort(grid_inds_1d)
    grid_inds_sorted_unique, lookup_inds_sorted, counts = torch.unique_consecutive(
        grid_inds_sorted, return_inverse=True, return_counts=True
    )
    del grid_inds_sorted
    lookup_struct = LookupStruct(
        grid_inds_sorted_unique, grid_inds_min, grid_inds_stride, dtype
    )
    lookup_inds = torch.empty_like(lookup_inds_sorted)
    lookup_inds = lookup_inds.index_copy_(0, grid_inds_sorter, lookup_inds_sorted)
    del lookup_inds_sorted

    return_list = [lookup_struct, lookup_inds]

    if return_unique_inds:
        unique_inds_sorted = torch.cumsum(counts, dim=0).sub_(1)
        unique_inds = grid_inds_sorter[unique_inds_sorted]
        return_list.append(unique_inds)
        del unique_inds_sorted

    if return_sorter:
        return_list.append(grid_inds_sorter)

    del grid_inds_sorter

    if return_unique_counts:
        return_list.append(counts)

    del counts

    return tuple(return_list)


def query_lookup_struct(lookup_struct, grid_inds):
    grid_inds_1d, _, _, _ = reduce_indices_to_1d(
        grid_inds,
        lookup_struct.grid_inds_min,
        lookup_struct.grid_inds_stride,
        lookup_struct.dtype,
    )
    lookup_inds = torch.searchsorted(
        lookup_struct.grid_inds_sorted_unique, grid_inds_1d
    )
    lookup_inds = lookup_inds.clamp_(max=lookup_struct.size() - 1)
    mask = lookup_struct.grid_inds_sorted_unique[lookup_inds].eq(grid_inds_1d)
    del grid_inds_1d
    lookup_inds = lookup_inds.mul_(mask)

    return lookup_inds, mask
