# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

import numpy as np

from ..grids import cropping_mask
from .dataset import Dataset
from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Masked(Forwards):
    def __init__(self, forward, mask):
        super().__init__(forward)
        assert len(forward.shape) == 4, "Grids must be 1D for now"
        self.mask = mask
        self.axis = 3

        self.mask_name = f"{self.__class__.__name__.lower()}_mask"

    @cached_property
    def shape(self):
        return self.forward.shape[:-1] + (np.count_nonzero(self.mask),)

    @cached_property
    def latitudes(self):
        return self.forward.latitudes[self.mask]

    @cached_property
    def longitudes(self):
        return self.forward.longitudes[self.mask]

    @debug_indexing
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._get_tuple(index)

        result = self.forward[index]
        # We don't support subsetting the grid values
        assert result.shape[-1] == len(self.mask), (result.shape, len(self.mask))

        return result[..., self.mask]

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, self.axis, slice(None))
        result = self.forward[index]
        result = result[..., self.mask]
        result = result[..., previous]
        result = apply_index_to_slices_changes(result, changes)
        return result

    def collect_supporting_arrays(self, collected, *path):
        super().collect_supporting_arrays(collected, *path)
        collected.append((path, self.mask_name, self.mask))


class Thinning(Masked):

    def __init__(self, forward, thinning, method):
        self.thinning = thinning
        self.method = method

        if thinning is not None:

            shape = forward.field_shape
            if len(shape) != 2:
                raise ValueError("Thinning only works latitude/longitude fields")

            # Make a copy, so we read the data only once from zarr
            forward_latitudes = forward.latitudes.copy()
            forward_longitudes = forward.longitudes.copy()

            latitudes = forward_latitudes.reshape(shape)
            longitudes = forward_longitudes.reshape(shape)
            latitudes = latitudes[::thinning, ::thinning].flatten()
            longitudes = longitudes[::thinning, ::thinning].flatten()

            # TODO: This is not very efficient

            mask = [lat in latitudes and lon in longitudes for lat, lon in zip(forward_latitudes, forward_longitudes)]
            mask = np.array(mask, dtype=bool)
        else:
            mask = None

        super().__init__(forward, mask)

    def mutate(self) -> Dataset:
        if self.thinning is None:
            return self.forward.mutate()
        return super().mutate()

    def tree(self):
        return Node(self, [self.forward.tree()], thinning=self.thinning, method=self.method)

    def subclass_metadata_specific(self):
        return dict(thinning=self.thinning, method=self.method)


class MaskFromDataset(Masked):

    def __init__(self, forward, dataset, field_name, threshold=0):
        from ..data import open_dataset

        self.dataset = open_dataset(dataset)
        if field_name not in self.dataset.dataset_metadata()["variables"]:
            raise ValueError(f"'{field_name}' is not a variable in the mask dataset")

        self.field_name = field_name
        self.threshold = threshold

        index = self.dataset.dataset_metadata()["variables"].index(field_name)
        mask = (self.dataset.data[0, index, 0, :] > threshold).astype(bool)

        super().__init__(forward, mask)

    def tree(self):
        return Node(
            self, [self.forward.tree()], dataset=self.dataset, field_name=self.field_name, threshold=self.threshold
        )

    def subclass_metadata_specific(self):
        return dict(dataset=self.dataset, field_name=self.field_name, threshold=self.threshold)

    @property
    def field_shape(self):
        return (np.sum(self.mask),)

    @property
    def grids(self):
        return (np.sum(self.mask),)


class Cropping(Masked):

    def __init__(self, forward, area):
        from ..data import open_dataset

        area = area if isinstance(area, (list, tuple)) else open_dataset(area)

        if isinstance(area, Dataset):
            north = np.amax(area.latitudes)
            south = np.amin(area.latitudes)
            east = np.amax(area.longitudes)
            west = np.amin(area.longitudes)
            area = (north, west, south, east)

        self.area = area
        mask = cropping_mask(forward.latitudes, forward.longitudes, *area)

        super().__init__(forward, mask)

    def tree(self):
        return Node(self, [self.forward.tree()], area=self.area)

    def subclass_metadata_specific(self):
        return dict(area=self.area)


class TrimEdge(Masked):

    def __init__(self, forward, edge):
        if isinstance(edge, int):
            self.edge = [edge] * 4
        elif len(edge) == 4:
            self.edge = edge
        else:
            raise ValueError("'edge' must be an integer or a list of 4 integers")

        shape = forward.field_shape
        if len(shape) != 2:
            raise ValueError("TrimEdge only works on regular grids")

        if self.edge[0] + self.edge[1] >= shape[0]:
            raise ValueError("Too much triming of the first grid dimension, resulting in an empty dataset")
        if self.edge[2] + self.edge[3] >= shape[1]:
            raise ValueError("Too much triming of the second grid dimension, resulting in an empty dataset")

        mask = np.full(shape, True, dtype=bool)
        mask[0:self.edge[0], :] = False
        mask[-self.edge[1]:, :] = False
        mask[:, 0:self.edge[2]] = False
        mask[:, -self.edge[3]:] = False

        mask = mask.flatten()

        super().__init__(forward, mask)

    def mutate(self) -> Dataset:
        if self.edge is None:
            return self.forward.mutate()
        return super().mutate()

    def tree(self):
        return Node(self, [self.forward.tree()], edge=self.edge)

    def subclass_metadata_specific(self):
        return dict(edge=self.edge)

    @property
    def field_shape(self):
        x, y = self.forward.field_shape
        x -= (self.edge[0] + self.edge[1])
        y -= (self.edge[2] + self.edge[3])
        return x, y
