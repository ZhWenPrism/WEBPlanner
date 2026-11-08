import pytest

from webplanner.evaluation import sizing_agreement, sizing_cases


def test_k_one_preserves_the_size_label_and_existing_agreement_column():
    report = sizing_cases([["4x2", "4x3"]], ["4x2"], k=1)
    assert report.loc[0, "top1"] == "4x2"
    assert bool(report.loc[0, "exact_top1"])
    agreement = sizing_agreement([["4x2", "4x3"]], ["4x2"], k=1)
    assert agreement["exact_top1"].k == 1
    assert "top1" not in agreement


def test_ranked_iterators_are_consumed_once():
    report = sizing_cases([iter(["4x3", "4x2"]), iter([4, 2])], ["4x2", "4x2"], k=2)
    assert report["top1"].tolist() == ["4x3", "4x2"]
    assert report["top2"].tolist() == [True, True]


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        sizing_cases(["4x2"], ["4x2"], k=k)
