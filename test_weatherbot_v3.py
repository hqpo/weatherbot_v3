import math
from weatherbot_v3 import (
    Book, Bucket, ForecastPoint, fractional_kelly,
    normalized_bucket_distribution, parse_bucket,
)


def test_regular_bucket_parse():
    assert parse_bucket("Will the highest temperature be between 82-83°F?", "F") == (82.0, 84.0)


def test_open_bucket_parse():
    assert parse_bucket("Will the highest temperature be 31°C or above?", "C") == (31.0, math.inf)


def test_distribution_sums_to_one():
    buckets = [
        Bucket("low", -math.inf, 29.5, "a", "ta"),
        Bucket("29", 29.5, 30.5, "b", "tb"),
        Bucket("high", 30.5, math.inf, "c", "tc"),
    ]
    probs = normalized_bucket_distribution(buckets, [ForecastPoint("x", 30.0, 1.0, 1.0)])
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["b"] > probs["a"]


def test_observed_max_eliminates_lower_bucket():
    buckets = [
        Bucket("28", 27.5, 28.5, "a", "ta"),
        Bucket("29", 28.5, 29.5, "b", "tb"),
        Bucket("30+", 29.5, math.inf, "c", "tc"),
    ]
    probs = normalized_bucket_distribution(buckets, [ForecastPoint("x", 29.0, 1.0, 1.0)], observed_max=29.6)
    assert probs["a"] == 0
    assert probs["b"] == 0
    assert probs["c"] == 1


def test_orderbook_vwap():
    book = Book([], [(0.20, 50), (0.25, 100)])
    shares, vwap, spent = book.buy_vwap(20)
    assert round(spent, 6) == 20
    assert round(shares, 6) == 90
    assert round(vwap, 6) == round(20 / 90, 6)


def test_kelly_is_conservative():
    assert 0 < fractional_kelly(0.40, 0.25, 0.15) < 0.1
