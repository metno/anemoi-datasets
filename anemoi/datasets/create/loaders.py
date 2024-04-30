# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import logging
import os
import time
import uuid
import warnings
from functools import cached_property

import numpy as np
import zarr

from anemoi.datasets import MissingDateError
from anemoi.datasets import open_dataset
from anemoi.datasets.create.persistent import build_storage
from anemoi.datasets.data.misc import as_first_date
from anemoi.datasets.data.misc import as_last_date
from anemoi.datasets.dates.groups import Groups

from .check import DatasetName
from .check import check_data_values
from .chunks import ChunkFilter
from .config import build_output
from .config import loader_config
from .input import build_input
from .statistics import Summary
from .statistics import TmpStatistics
from .statistics import check_variance
from .statistics import compute_statistics
from .statistics import default_statistics_dates
from .utils import normalize_and_check_dates
from .utils import progress_bar
from .utils import seconds
from .writer import ViewCacheArray
from .zarr import ZarrBuiltRegistry
from .zarr import add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"


class GenericDatasetHandler:
    def __init__(self, *, path, print=print, **kwargs):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        assert isinstance(path, str), path

        self.path = path
        self.kwargs = kwargs
        self.print = print

    @classmethod
    def from_config(cls, *, config, path, print=print, **kwargs):
        # config is the path to the config file or a dict with the config
        assert isinstance(config, dict) or isinstance(config, str), config
        return cls(config=config, path=path, print=print, **kwargs)

    @classmethod
    def from_dataset_config(cls, *, path, print=print, **kwargs):
        assert os.path.exists(path), f"Path {path} does not exist."
        z = zarr.open(path, mode="r")
        config = z.attrs["_create_yaml_config"]
        LOG.info(f"Config loaded from zarr config: {config}")
        return cls.from_config(config=config, path=path, print=print, **kwargs)

    @classmethod
    def from_dataset(cls, *, path, **kwargs):
        assert os.path.exists(path), f"Path {path} does not exist."
        return cls(path=path, **kwargs)

    def read_dataset_metadata(self):
        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables
        assert len(self.variables_names) == ds.shape[1], self.dataset_shape
        self.dates = ds.dates

        self.missing_dates = sorted(list([self.dates[i] for i in ds.missing]))

        z = zarr.open(self.path, "r")
        missing_dates = z.attrs.get("missing_dates", [])
        missing_dates = sorted([np.datetime64(d) for d in missing_dates])
        assert missing_dates == self.missing_dates, (missing_dates, self.missing_dates)

    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path)

    def update_metadata(self, **kwargs):
        LOG.info(f"Updating metadata {kwargs}")
        z = zarr.open(self.path, mode="w+")
        for k, v in kwargs.items():
            if isinstance(v, np.datetime64):
                v = v.astype(datetime.datetime)
            if isinstance(v, datetime.date):
                v = v.isoformat()
            z.attrs[k] = v

    def _add_dataset(self, mode="r+", **kwargs):
        z = zarr.open(self.path, mode=mode)
        return add_zarr_dataset(zarr_root=z, **kwargs)

    def get_zarr_chunks(self):
        z = zarr.open(self.path, mode="r")
        return z["data"].chunks

    def print_info(self):
        z = zarr.open(self.path, mode="r")
        try:
            LOG.info(z["data"].info)
        except Exception as e:
            LOG.info(e)


class DatasetHandler(GenericDatasetHandler):
    pass


class DatasetHandlerWithStatistics(GenericDatasetHandler):
    def __init__(self, statistics_tmp=None, **kwargs):
        super().__init__(**kwargs)
        statistics_tmp = kwargs.get("statistics_tmp") or os.path.join(self.path + ".tmp_data", "statistics")
        self.tmp_statistics = TmpStatistics(statistics_tmp)


class Loader(DatasetHandlerWithStatistics):
    def build_input(self):
        from climetlab.core.order import build_remapping

        builder = build_input(
            self.main_config.input,
            data_sources=self.main_config.get("data_sources", {}),
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
        )
        LOG.info("✅ INPUT_BUILDER")
        LOG.info(builder)
        return builder

    def allow_nan(self, name):
        return name in self.main_config.statistics.get("allow_nans", [])


class InitialiserLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.tmp_statistics.delete()

        LOG.info(self.main_config.dates)
        self.groups = Groups(**self.main_config.dates)

        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()

        LOG.info(self.input)
        all_dates = self.groups.dates
        self.minimal_input = self.input.select([all_dates[0]])

        LOG.info(self.groups)
        LOG.info("MINIMAL INPUT :")
        LOG.info(self.minimal_input)

    def build_statistics_dates(self, start, end):
        ds = open_dataset(self.path)
        dates = ds.dates

        default_start, default_end = default_statistics_dates(dates)
        if start is None:
            start = default_start
        if end is None:
            end = default_end

        start = as_first_date(start, dates)
        end = as_last_date(end, dates)

        start = start.astype(datetime.datetime)
        end = end.astype(datetime.datetime)
        return (start.isoformat(), end.isoformat())

    def initialise_dataset_backend(self):
        z = zarr.open(self.path, mode="w")
        z.create_group("_build")

    def initialise(self, check_name=True):
        """Create empty dataset."""

        self.print("Config loaded ok:")
        LOG.info(self.main_config)

        dates = self.groups.dates
        frequency = dates.frequency
        assert isinstance(frequency, int), frequency

        self.print(f"Found {len(dates)} datetimes.")
        LOG.info(f"Dates: Found {len(dates)} datetimes, in {len(self.groups)} groups: ")
        LOG.info(f"Missing dates: {len(dates.missing)}")
        lengths = [len(g) for g in self.groups]
        self.print(f"Found {len(dates)} datetimes {'+'.join([str(_) for _ in lengths])}.")

        variables = self.minimal_input.variables
        self.print(f"Found {len(variables)} variables : {','.join(variables)}.")

        variables_with_nans = self.main_config.statistics.get("allow_nans", [])

        ensembles = self.minimal_input.ensembles
        self.print(f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}.")

        grid_points = self.minimal_input.grid_points
        LOG.info(f"gridpoints size: {[len(i) for i in grid_points]}")

        resolution = self.minimal_input.resolution
        LOG.info(f"{resolution=}")

        coords = self.minimal_input.coords
        coords["dates"] = dates
        total_shape = self.minimal_input.shape
        total_shape[0] = len(dates)
        self.print(f"total_shape = {total_shape}")

        chunks = self.output.get_chunking(coords)
        LOG.info(f"{chunks=}")
        dtype = self.output.dtype

        self.print(f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}")

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        metadata.update(self.main_config.get("add_metadata", {}))

        metadata["_create_yaml_config"] = self.main_config.get_serialisable_dict()

        metadata["description"] = self.main_config.description
        metadata["version"] = VERSION

        metadata["data_request"] = self.minimal_input.data_request
        metadata["remapping"] = self.output.remapping

        metadata["order_by"] = self.output.order_by_as_list
        metadata["flatten_grid"] = self.output.flatten_grid

        metadata["ensemble_dimension"] = len(ensembles)
        metadata["variables"] = variables
        metadata["variables_with_nans"] = variables_with_nans
        metadata["resolution"] = resolution
        metadata["field_shape"] = self.minimal_input.field_shape
        metadata["proj_string"] = self.minimal_input.proj_string

        metadata["licence"] = self.main_config["licence"]
        metadata["attribution"] = self.main_config["attribution"]

        metadata["frequency"] = frequency
        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()
        metadata["missing_dates"] = [_.isoformat() for _ in dates.missing]

        if check_name:
            basename, ext = os.path.splitext(os.path.basename(self.path))  # noqa: F841
            ds_name = DatasetName(
                basename,
                resolution,
                dates[0],
                dates[-1],
                frequency,
            )
            ds_name.raise_if_not_valid(print=self.print)

        if len(dates) != total_shape[0]:
            raise ValueError(
                f"Final date size {len(dates)} (from {dates[0]} to {dates[-1]}, {frequency=}) "
                f"does not match data shape {total_shape[0]}. {total_shape=}"
            )

        dates = normalize_and_check_dates(
            dates,
            metadata["start_date"],
            metadata["end_date"],
            metadata["frequency"],
        )

        metadata.update(self.main_config.get("force_metadata", {}))

        ###############################################################
        # write data
        ###############################################################

        self.initialise_dataset_backend()

        self.update_metadata(**metadata)

        self._add_dataset(name="data", chunks=chunks, dtype=dtype, shape=total_shape)
        self._add_dataset(name="dates", array=dates)
        self._add_dataset(name="latitudes", array=grid_points[0])
        self._add_dataset(name="longitudes", array=grid_points[1])

        self.registry.create(lengths=lengths)
        self.tmp_statistics.create(exist_ok=False)
        self.registry.add_to_history("tmp_statistics_initialised", version=self.tmp_statistics.version)

        statistics_start, statistics_end = self.build_statistics_dates(
            self.main_config.statistics.get("start"),
            self.main_config.statistics.get("end"),
        )
        self.update_metadata(
            statistics_start_date=statistics_start,
            statistics_end_date=statistics_end,
        )
        LOG.info(f"Will compute statistics from {statistics_start} to {statistics_end}")

        self.registry.add_to_history("init finished")

        assert chunks == self.get_zarr_chunks(), (chunks, self.get_zarr_chunks())


class ContentLoader(Loader):
    def __init__(self, config, parts, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.groups = Groups(**self.main_config.dates)
        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()
        self.read_dataset_metadata()

        self.parts = parts
        total = len(self.registry.get_flags())
        self.chunk_filter = ChunkFilter(parts=self.parts, total=total)

        self.data_array = zarr.open(self.path, mode="r+")["data"]
        self.n_groups = len(self.groups)

    def load(self):
        self.registry.add_to_history("loading_data_start", parts=self.parts)

        for igroup, group in enumerate(self.groups):
            if not self.chunk_filter(igroup):
                continue
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue
            # self.print(f" -> Processing {igroup} total={len(self.groups)}")
            # print("========", group)
            assert isinstance(group[0], datetime.datetime), group

            result = self.input.select(dates=group)
            assert result.dates == group, (len(result.dates), len(group))

            msg = f"Building data for group {igroup}/{self.n_groups}"
            LOG.info(msg)
            self.print(msg)

            # There are several groups.
            # There is one result to load for each group.
            self.load_result(result)
            self.registry.set_flag(igroup)

        self.registry.add_to_history("loading_data_end", parts=self.parts)
        self.registry.add_provenance(name="provenance_load")
        self.tmp_statistics.add_provenance(name="provenance_load", config=self.main_config)

        self.print_info()

    def load_result(self, result):
        # There is one cube to load for each result.
        dates = result.dates

        cube = result.get_cube()
        assert cube.extended_user_shape[0] == len(dates), (
            cube.extended_user_shape[0],
            len(dates),
        )

        shape = cube.extended_user_shape
        dates_in_data = cube.user_coords["valid_datetime"]

        LOG.info(f"Loading {shape=} in {self.data_array.shape=}")

        def check_dates_in_data(lst, lst2):
            lst2 = [np.datetime64(_) for _ in lst2]
            lst = [np.datetime64(_) for _ in lst]
            assert lst == lst2, ("Dates in data are not the requested ones:", lst, lst2)

        check_dates_in_data(dates_in_data, dates)

        def dates_to_indexes(dates, all_dates):
            x = np.array(dates, dtype=np.datetime64)
            y = np.array(all_dates, dtype=np.datetime64)
            bitmap = np.isin(x, y)
            return np.where(bitmap)[0]

        indexes = dates_to_indexes(self.dates, dates_in_data)

        array = ViewCacheArray(self.data_array, shape=shape, indexes=indexes)
        self.load_cube(cube, array)

        stats = compute_statistics(array.cache, self.variables_names, allow_nan=self.allow_nan)
        self.tmp_statistics.write(indexes, stats, dates=dates_in_data)

        array.flush()

    def load_cube(self, cube, array):
        # There are several cubelets for each cube
        start = time.time()
        load = 0
        save = 0

        reading_chunks = None
        total = cube.count(reading_chunks)
        self.print(f"Loading datacube: {cube}")
        bar = progress_bar(
            iterable=cube.iterate_cubelets(reading_chunks),
            total=total,
            desc=f"Loading datacube {cube}",
        )
        for i, cubelet in enumerate(bar):
            bar.set_description(f"Loading {i}/{total}")

            now = time.time()
            data = cubelet.to_numpy()
            local_indexes = cubelet.coords
            load += time.time() - now

            name = self.variables_names[local_indexes[1]]
            check_data_values(
                data[:],
                name=name,
                log=[i, data.shape, local_indexes],
                allow_nan=self.allow_nan,
            )

            now = time.time()
            array[local_indexes] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now
        LOG.info("Written.")
        msg = f"Elapsed: {seconds(time.time() - start)}, load time: {seconds(load)}, write time: {seconds(save)}."
        self.print(msg)
        LOG.info(msg)


class StatisticsAdder(DatasetHandlerWithStatistics):
    def __init__(
        self,
        statistics_output=None,
        statistics_start=None,
        statistics_end=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_statistics_start = statistics_start
        self.user_statistics_end = statistics_end

        self.statistics_output = statistics_output

        self.output_writer = {
            None: self.write_stats_to_dataset,
            "-": self.write_stats_to_stdout,
        }.get(self.statistics_output, self.write_stats_to_file)

        self.read_dataset_metadata()

    def allow_nan(self, name):
        z = zarr.open(self.path, mode="r")
        if "variables_with_nans" in z.attrs:
            return name in z.attrs["variables_with_nans"]

        warnings.warn(f"Cannot find 'variables_with_nans' in {self.path}. Assuming nans allowed for {name}.")
        return True

    def _get_statistics_dates(self):
        dates = self.dates
        dtype = type(dates[0])

        def assert_dtype(d):
            assert type(d) is dtype, (type(d), dtype)

        # remove missing dates
        if self.missing_dates:
            assert_dtype(self.missing_dates[0])
        dates = [d for d in dates if d not in self.missing_dates]

        # filter dates according the the start and end dates in the metadata
        z = zarr.open(self.path, mode="r")
        start, end = z.attrs.get("statistics_start_date"), z.attrs.get("statistics_end_date")
        start, end = np.datetime64(start), np.datetime64(end)
        assert_dtype(start)
        dates = [d for d in dates if d >= start and d <= end]

        # filter dates according the the user specified start and end dates
        if self.user_statistics_start:
            limit = as_first_date(self.user_statistics_start, dates)
            limit = np.datetime64(limit)
            assert_dtype(limit)
            dates = [d for d in dates if d >= limit]

        if self.user_statistics_end:
            limit = as_last_date(self.user_statistics_start, dates)
            limit = np.datetime64(limit)
            assert_dtype(limit)
            dates = [d for d in dates if d <= limit]

        return dates

    def run(self):
        dates = self._get_statistics_dates()
        stats = self.tmp_statistics.get_aggregated(dates, self.variables_names, self.allow_nan)
        self.output_writer(stats)

    def write_stats_to_file(self, stats):
        stats.save(self.statistics_output)
        LOG.info(f"✅ Statistics written in {self.statistics_output}")

    def write_stats_to_dataset(self, stats):
        if self.user_statistics_start or self.user_statistics_end:
            raise ValueError(
                (
                    "Cannot write statistics in dataset with user specified dates. "
                    "This would be conflicting with the dataset metadata."
                )
            )

        if not all(self.registry.get_flags(sync=False)):
            raise Exception(f"❗Zarr {self.path} is not fully built, not writting statistics into dataset.")

        for k in [
            "mean",
            "stdev",
            "minimum",
            "maximum",
            "sums",
            "squares",
            "count",
            "has_nans",
        ]:
            self._add_dataset(name=k, array=stats[k])

        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote statistics in {self.path}")

    def write_stats_to_stdout(self, stats):
        LOG.info(stats)


class GenericAdditionsLoader(GenericDatasetHandler):
    def __init__(self, name="", **kwargs):
        super().__init__(**kwargs)
        self.name = name

        assert len(self.ds.variables) == self.ds.shape[1], self.ds.shape
        self.dates = self.ds.dates

        storage_path = os.path.join(self.path + ".tmp_data", name)
        self.tmp_storage = build_storage(directory=storage_path, create=True)

    def initialise(self):
        self.tmp_storage.delete()
        self.tmp_storage.create()
        LOG.info(f"Dataset {self.path} additions initialized.")

    @cached_property
    def ds(self):
        return open_dataset(self.path, start=self.start, end=self.end)

    @cached_property
    def _variables_with_nans(self):
        z = zarr.open(self.path, mode="r")
        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]
        return None

    def allow_nan(self, name):
        if self._variables_with_nans is not None:
            return name in self._variables_with_nans
        warnings.warn(f"❗Cannot find 'variables_with_nans' in {self.path}, Assuming nans allowed for {name}.")
        return True

    @classmethod
    def _check_type_equal(cls, a, b):
        a = list(a)
        b = list(b)
        a = a[0] if a else None
        b = b[0] if b else None
        assert type(a) is type(b), (type(a), type(b))

    def _get_empty_stats_dict(self, shape):
        return dict(
            minimum=np.full(self.shape, np.nan, dtype=np.float64),
            maximum=np.full(self.shape, np.nan, dtype=np.float64),
            sums=np.full(self.shape, np.nan, dtype=np.float64),
            squares=np.full(self.shape, np.nan, dtype=np.float64),
            count=np.full(self.shape, -1, dtype=np.int64),
            has_nans=np.full(self.shape, False, dtype=np.bool_),
        )

    def finalise(self):
        self.shape = (len(self.dates), len(self.ds.variables))
        LOG.info(f"Aggregating {self.name} statistics on shape={self.shape}. Variables : {self.ds.variables}")

        agg = self._get_empty_stats_dict(self.shape)

        found = set()
        missing = set()
        mask = []
        for _dates, (dates, missing_dates, _, stats) in self.tmp_storage.items():
            if missing_dates:
                mask.append(False)
                missing |= set(missing_dates)
                continue
            mask.append(True)

            assert isinstance(stats, dict), stats
            shape = stats["minimum"].shape
            assert shape[0] == len(dates), (shape, len(dates))
            assert shape[0] == len(_dates), (shape, len(_dates))
            assert shape[1] == len(self.ds.variables), (shape, len(self.ds.variables))

            assert found.isdisjoint(dates), "Duplicates found"
            found |= set(dates)

            self._check_type_equal(dates, _dates)
            idates = [i for i, d in enumerate(self.dates) if d in dates]
            assert len(idates) == len(dates), (len(idates), len(dates))

            for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
                agg[k][idates, ...] = stats[k]

        for d in self.dates:
            assert d in found or d in missing, f"Statistics for date {d} not precomputed."
        assert len(self.dates) == len(found) + len(missing), "Not all dates found in precomputed statistics"
        assert found.isdisjoint(missing), "Duplicate dates found in precomputed statistics"
        assert agg["minimum"].shape[0] == len(self.dates), (agg["minimum"].shape, len(self.dates))

        for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            agg[k] = agg[k][mask, ...]
        assert agg["minimum"].shape[0] == len(self.dates) - len(missing), (
            agg["minimum"].shape,
            len(self.dates),
            len(missing),
        )

        minimum = np.nanmin(agg["minimum"], axis=0)
        maximum = np.nanmax(agg["maximum"], axis=0)
        sums = np.nansum(agg["sums"], axis=0)
        squares = np.nansum(agg["squares"], axis=0)
        count = np.nansum(agg["count"], axis=0)
        has_nans = np.any(agg["has_nans"], axis=0)

        assert sums.shape == count.shape
        assert sums.shape == squares.shape
        assert sums.shape == minimum.shape
        assert sums.shape == maximum.shape
        assert sums.shape == has_nans.shape

        mean = sums / count
        assert sums.shape == mean.shape

        x = squares / count - mean * mean
        # remove negative variance due to numerical errors
        # x[- 1e-15 < (x / (np.sqrt(squares / count) + np.abs(mean))) < 0] = 0
        check_variance(x, self.ds.variables, minimum, maximum, mean, count, sums, squares)

        stdev = np.sqrt(x)
        assert sums.shape == stdev.shape

        self.summary = Summary(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            count=count,
            sums=sums,
            squares=squares,
            stdev=stdev,
            variables_names=self.ds.variables,
            has_nans=has_nans,
        )
        LOG.info(f"Dataset {self.path} additions finalized.")
        self.check_statistics()
        self._write(self.summary)
        self.tmp_storage.delete()

    def _write(self, summary):
        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            self._add_dataset(name=k, array=summary[k])
        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote {self.name} additions in {self.path}")

    def check_statistics(self):
        ds = open_dataset(self.path)
        ref = ds.statistics
        for k in ds.statistics:
            assert np.all(np.isclose(ref[k], self.summary[k], rtol=1e-4, atol=1e-4)), (
                k,
                ref[k],
                self.summary[k],
            )


class AdditionsLoader(GenericAdditionsLoader):

    @property
    def total(self):
        return len(self.ds.dates)

    @cached_property
    def input_array(self):
        return self.ds

    @cached_property
    def start(self):
        z = zarr.open(self.path, mode="r")
        return z.attrs["statistics_start_date"]

    @cached_property
    def end(self):
        z = zarr.open(self.path, mode="r")
        return z.attrs["statistics_end_date"]

    def run(self, parts):
        chunk_filter = ChunkFilter(parts=parts, total=self.total)
        for i in range(0, self.total):
            if not chunk_filter(i):
                continue
            date = self.dates[i]
            # self.print(f"{self.name} additions : Processing {date} ({i+1}/{self.total})")
            try:
                arr = self.input_array[i : i + 1, ...]
                stats = compute_statistics(arr, self.ds.variables, allow_nan=self.allow_nan)
                dates_ok = [date]
                missing_dates = []
            except MissingDateError:
                stats = None
                dates_ok = []
                missing_dates = [date]

            dates = sorted(dates_ok + missing_dates)
            self.tmp_storage.add([dates_ok, missing_dates, i, stats], key=dates)
        self.tmp_storage.flush()

        LOG.info(f"Dataset {self.path} additions run.")
