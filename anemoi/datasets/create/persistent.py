# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import glob
import hashlib
import json
import logging
import os
import pickle
import shutil
import socket

import numpy as np
from anemoi.utils.provenance import gather_provenance_info

LOG = logging.getLogger(__name__)


class PersistentDict:
    version = 3

    # Used in parrallel, during data loading,
    # to write data in pickle files.
    def __init__(self, directory, create=True):
        """dirname: str
        The directory where the data will be stored.
        """
        self.dirname = directory
        self.name, self.ext = os.path.splitext(os.path.basename(self.dirname))
        if create:
            os.makedirs(self.dirname, exist_ok=True)

    def delete(self):
        try:
            shutil.rmtree(self.dirname)
        except FileNotFoundError:
            pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.dirname})"

    def items(self):
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.pickle")
        LOG.info(f"Reading {self.name} data, found {len(files)} files in {self.dirname}")
        assert len(files) > 0, f"No files found in {self.dirname}"
        for f in files:
            with open(f, "rb") as f:
                yield pickle.load(f)

    def add_provenance(self, **kwargs):
        out = dict(provenance=gather_provenance_info(), **kwargs)
        with open(os.path.join(self.dirname, "provenance.json"), "w") as f:
            json.dump(out, f)

    def __setitem__(self, key, elt):
        h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        path = os.path.join(self.dirname, f"{h}.pickle")

        if os.path.exists(path):
            LOG.warn(f"{path} already exists")

        tmp_path = path + f".tmp-{os.getpid()}-on-{socket.gethostname()}"
        with open(tmp_path, "wb") as f:
            pickle.dump((key, elt), f)
        shutil.move(tmp_path, path)

        LOG.debug(f"Written {self.name} data for len {key} in {path}")


if __name__ == "__main__":
    N = 3
    P = 2
    directory = "h"
    p = PersistentDict(directory=directory)
    print(p)
    assert os.path.exists(directory)
    import numpy as np

    arrs = [np.random.randint(1, 101, size=(P,)) for _ in range(N)]
    dates = [np.array([np.datetime64(f"2021-01-0{_+1}") + np.timedelta64(i, "h") for i in range(P)]) for _ in range(N)]

    print()
    print("Writing the data")
    for i in range(N):
        _arr = arrs[i]
        _dates = dates[i]
        print(f"Writing : {i=}, {_arr=} {_dates=}")
        p[_dates] = (i, _arr)

    print()
    print("Reading the data back")

    p = PersistentDict(directory="h")
    for _dates, (i, _arr) in p.items():
        print(f"{i=}, {_arr=}, {_dates=}")

        assert np.allclose(_arr, arrs[i])

        assert len(_dates) == len(dates[i])
        for a, b in zip(_dates, dates[i]):
            assert a == b
