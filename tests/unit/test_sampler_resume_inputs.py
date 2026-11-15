from itertools import islice

import numpy as np
import pytest

from webplanner.data.dataset import InfiniteSampler


@pytest.mark.parametrize(("n", "start", "name"), [(0.5, 0, "n"), (True, 0, "n"),
                                                (3, -1, "start"), (3, 2.5, "start")])
def test_sampler_rejects_invalid_case_counts_and_resume_positions(n, start, name):
    with pytest.raises(ValueError, match=rf"^{name} must"):
        InfiniteSampler(n, start=start)


def test_resumed_sampler_continues_across_a_round_boundary():
    entire = list(islice(InfiniteSampler(4, seed=5), 13))
    resumed = list(islice(InfiniteSampler(np.int64(4), seed=5, start=np.int64(3)), 10))
    assert resumed == entire[3:]
