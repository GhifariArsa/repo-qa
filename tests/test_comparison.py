"""The comparison sampler must fill the rare outcome, not just the common one."""

import random

from cgx.qgen.comparison import _candidate_pairs


def _rows():
    # 200 callables, each calling its own private helper; exactly one pair
    # (f0, f1) shares a callee, and f2 calls a strict superset of f3's callees.
    rows = [{"qname": f"f{i:03d}", "members": [f"h{i}"]} for i in range(200)]
    rows[0]["members"] = ["h0", "shared"]
    rows[1]["members"] = ["h1", "shared"]
    rows[2]["members"] = ["h2", "h3"]
    return rows


def test_intersection_finds_the_one_shared_pair():
    pairs = _candidate_pairs(_rows(), limit=4, rng=random.Random(0))
    assert ("f000", "f001") in pairs


def test_difference_finds_an_empty_difference():
    pairs = _candidate_pairs(_rows(), limit=4, rng=random.Random(0), difference=True)
    assert ("f003", "f002") in pairs      # {h3} - {h2, h3} is empty


def test_odd_quota_extra_goes_to_either_side():
    """Always giving an odd count's extra to "yes" tilted the pooled set 10:1."""
    from cgx.qgen._util import split_quota

    splits = [split_quota(1, random.Random(i)) for i in range(200)]
    assert all(sum(s) == 1 for s in splits)
    yes_share = sum(s[0] for s in splits) / len(splits)
    assert 0.35 < yes_share < 0.65
    assert split_quota(4, random.Random(0)) == (2, 2)
