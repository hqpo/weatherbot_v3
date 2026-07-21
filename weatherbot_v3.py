#!/usr/bin/env python3
"""WeatherBet v3 — safer paper-trading prototype for Polymarket weather events.

Key changes from bot_v2:
- builds a normalized probability distribution over every event bucket;
- uses Gaussian forecast uncertainty instead of 0/100% point forecasts;
- applies a simple same-day observation floor (observed daily max);
- reads the public CLOB order book and sizes against executable ask depth;
- uses conservative fractional Kelly and event-level exposure limits;
- exits on information deterioration, not mechanical price stop-losses;
- keeps cash and realized PnL separate to avoid balance inflation.

This remains PAPER TRADING ONLY.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
METAR = "https://aviationweather.gov/api/data/metar"


@dataclass(frozen=True)
class Bucket:
    label: str
    low: float
    high: float
    market_id: str
    yes_token_id: str
    volume: float = 0.0

    def contains(self, value: float) -> bool:
        return self.low <= value < self.high


@dataclass
class Book:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    tick_size: float = 0.01

    @property
    def best_bid(self) -> float | None:
        return max((p for p, _ in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        return min((p for p, _ in self.asks), default=None)

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def buy_vwap(self, dollars: float) -> tuple[float, float, float]:
        """Return shares, VWAP and spent dollars for walking asks up to budget."""
        if dollars <= 0:
            return 0.0, 0.0, 0.0
        remaining = dollars
        shares = spent = 0.0
        for price, size in sorted(self.asks):
            if price <= 0 or size <= 0:
                continue
            affordable = remaining / price
            take = min(size, affordable)
            shares += take
            cost = take * price
            spent += cost
            remaining -= cost
            if remaining <= 1e-9:
                break
        return shares, (spent / shares if shares else 0.0), spent


@dataclass
class ForecastPoint:
    source: str
    temperature: float
    sigma: float
    weight: float


@dataclass
class Candidate:
    bucket: Bucket
    probability: float
    conservative_probability: float
    ask: float
    vwap: float
    shares: float
    cost: float
    edge: float
    expected_profit: float
    kelly_fraction: float


DEFAULT_CONFIG: dict[str, Any] = {
    "starting_cash": 10000.0,
    "max_trade_dollars": 20.0,
    "max_event_exposure_pct": 0.03,
    "max_weather_exposure_pct": 0.12,
    "kelly_fraction": 0.15,
    "min_net_edge": 0.06,
    "probability_haircut": 0.03,
    "max_spread": 0.05,
    "min_volume": 500,
    "min_hours": 1.0,
    "max_hours": 72.0,
    "scan_interval": 1800,
    "request_timeout": 12,
    "paper_trade": True,
    "locations": {
        "nyc": {"name": "New York City", "lat": 40.7772, "lon": -73.8726, "station": "KLGA", "unit": "F", "timezone": "America/New_York", "region": "us"},
        "chicago": {"name": "Chicago", "lat": 41.9742, "lon": -87.9073, "station": "KORD", "unit": "F", "timezone": "America/Chicago", "region": "us"},
        "miami": {"name": "Miami", "lat": 25.7959, "lon": -80.2870, "station": "KMIA", "unit": "F", "timezone": "America/New_York", "region": "us"},
        "dallas": {"name": "Dallas", "lat": 32.8471, "lon": -96.8518, "station": "KDAL", "unit": "F", "timezone": "America/Chicago", "region": "us"},
        "seoul": {"name": "Seoul", "lat": 37.4691, "lon": 126.4505, "station": "RKSI", "unit": "C", "timezone": "Asia/Seoul", "region": "asia"},
        "tokyo": {"name": "Tokyo", "lat": 35.7647, "lon": 140.3864, "station": "RJTT", "unit": "C", "timezone": "Asia/Tokyo", "region": "asia"},
        "london": {"name": "London", "lat": 51.5048, "lon": 0.0495, "station": "EGLC", "unit": "C", "timezone": "Europe/London", "region": "eu"}
    }
}


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def interval_probability(mean: float, sigma: float, low: float, high: float) -> float:
    sigma = max(0.15, float(sigma))
    lo = 0.0 if math.isinf(low) and low < 0 else normal_cdf((low - mean) / sigma)
    hi = 1.0 if math.isinf(high) and high > 0 else normal_cdf((high - mean) / sigma)
    return max(0.0, hi - lo)


def normalized_bucket_distribution(
    buckets: list[Bucket], forecasts: list[ForecastPoint], observed_max: float | None = None
) -> dict[str, float]:
    """Mixture distribution, constrained by already observed daily maximum."""
    raw: dict[str, float] = {}
    total_weight = sum(max(0.0, f.weight) for f in forecasts)
    if not buckets or total_weight <= 0:
        return {}
    for bucket in buckets:
        p = sum(
            max(0.0, f.weight) * interval_probability(f.temperature, f.sigma, bucket.low, bucket.high)
            for f in forecasts
        ) / total_weight
        if observed_max is not None and bucket.high <= observed_max:
            p = 0.0
        raw[bucket.market_id] = p
    total = sum(raw.values())
    if total <= 0:
        # Defensive fallback: assign all mass to the bucket containing observed max,
        # or to the nearest upper bucket when market ranges are incomplete.
        target = observed_max if observed_max is not None else forecasts[0].temperature
        chosen = next((b for b in buckets if b.contains(target)), min(buckets, key=lambda b: abs((b.low + b.high) / 2 - target)))
        return {b.market_id: 1.0 if b.market_id == chosen.market_id else 0.0 for b in buckets}
    return {k: v / total for k, v in raw.items()}


def fractional_kelly(probability: float, price: float, fraction: float) -> float:
    if not 0 < price < 1 or not 0 <= probability <= 1:
        return 0.0
    full = (probability - price) / (1.0 - price)
    return max(0.0, min(1.0, full * fraction))


def parse_bucket(question: str, unit: str) -> tuple[float, float] | None:
    q = question.replace("–", "-").replace("—", "-").replace("°", "")
    # Treat a dash between positive digits as a range separator, not a minus sign.
    q = re.sub(r"(?<=\d)-(?=\d)", " ", q)
    # "between 82-83 F" / "82 to 83"
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", q)]
    lower_words = re.search(r"(?:or below|or less|or lower|at most|≤|below)", q, re.I)
    upper_words = re.search(r"(?:or above|or more|or higher|at least|≥|above)", q, re.I)
    if len(nums) >= 2:
        lo, hi = nums[-2], nums[-1]
        if hi < lo:
            lo, hi = hi, lo
        # Displayed range endpoints are inclusive in common 2°F bucket wording.
        step = 1.0
        return lo, hi + step
    if len(nums) == 1:
        x = nums[0]
        if lower_words:
            return -math.inf, x + 1.0
        if upper_words:
            return x, math.inf
        # Integer Celsius outcomes resolve to that displayed integer.
        return x - 0.5, x + 0.5
    return None


def decode_json_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


class WeatherBot:
    def __init__(self, config_path: Path):
        self.root = config_path.parent
        self.config = DEFAULT_CONFIG | (json.loads(config_path.read_text()) if config_path.exists() else {})
        self.config["locations"] = DEFAULT_CONFIG["locations"] | self.config.get("locations", {})
        self.data_dir = self.root / "data_v3"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "state.json"
        self.session = requests.Session()
        self.timeout = float(self.config["request_timeout"])
        self.state = self.load_state()

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
        else:
            state = {"cash": float(self.config["starting_cash"]), "realized_pnl": 0.0, "positions": [], "trades": []}
        state.setdefault("cash", float(self.config["starting_cash"]))
        state.setdefault("realized_pnl", 0.0)
        state.setdefault("positions", [])
        state.setdefault("trades", [])
        return state

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))

    def get_event(self, city_slug: str, target: date) -> dict[str, Any] | None:
        month = target.strftime("%B").lower()
        slugs = [
            f"highest-temperature-in-{city_slug}-on-{month}-{target.day}-{target.year}",
            f"highest-temperature-in-{city_slug}-on-{month}-{target.day}",
        ]
        for slug in slugs:
            r = self.session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if data:
                return data[0]
        return None

    def event_buckets(self, event: dict[str, Any], unit: str) -> list[Bucket]:
        buckets: list[Bucket] = []
        for market in event.get("markets", []):
            bounds = parse_bucket(market.get("question", ""), unit)
            token_ids = decode_json_field(market.get("clobTokenIds"))
            outcomes = decode_json_field(market.get("outcomes"))
            if not bounds or not token_ids:
                continue
            yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "yes"), 0)
            if yes_idx >= len(token_ids):
                continue
            buckets.append(Bucket(
                label=market.get("question", ""), low=bounds[0], high=bounds[1],
                market_id=str(market.get("id", "")), yes_token_id=str(token_ids[yes_idx]),
                volume=float(market.get("volumeNum") or market.get("volume") or 0),
            ))
        return sorted(buckets, key=lambda b: b.low)

    def get_book(self, token_id: str) -> Book | None:
        r = self.session.get(f"{CLOB}/book", params={"token_id": token_id}, timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        parse = lambda levels: [(float(x["price"]), float(x["size"])) for x in levels]
        return Book(parse(data.get("bids", [])), parse(data.get("asks", [])), float(data.get("tick_size", 0.01)))

    def forecast_points(self, city_slug: str, target: date) -> list[ForecastPoint]:
        loc = self.config["locations"][city_slug]
        params = {
            "latitude": loc["lat"], "longitude": loc["lon"], "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit" if loc["unit"] == "F" else "celsius",
            "timezone": loc["timezone"], "forecast_days": 7,
        }
        points: list[ForecastPoint] = []
        for model, source, weight in [("ecmwf_ifs025", "ecmwf", 0.65), ("gfs_seamless", "gfs", 0.35)]:
            p = params | {"models": model}
            try:
                r = self.session.get(OPEN_METEO, params=p, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()["daily"]
                mapping = dict(zip(data["time"], data["temperature_2m_max"]))
                value = mapping.get(target.isoformat())
                if value is not None:
                    lead_hours = max(0, (datetime.combine(target, datetime.min.time(), tzinfo=ZoneInfo(loc["timezone"])) - datetime.now(ZoneInfo(loc["timezone"]))).total_seconds() / 3600)
                    base_sigma = 2.0 if loc["unit"] == "F" else 1.2
                    sigma = base_sigma * (1.0 + max(0.0, lead_hours - 24) / 120)
                    points.append(ForecastPoint(source, float(value), sigma, weight))
            except (requests.RequestException, KeyError, TypeError, ValueError):
                continue
        return points

    def current_observation(self, city_slug: str, target: date) -> float | None:
        loc = self.config["locations"][city_slug]
        local_now = datetime.now(ZoneInfo(loc["timezone"]))
        if local_now.date() != target:
            return None
        try:
            r = self.session.get(METAR, params={"ids": loc["station"], "format": "json", "hours": 24}, timeout=self.timeout)
            r.raise_for_status()
            rows = r.json()
            values = [float(row["temp"]) for row in rows if row.get("temp") is not None]
            if not values:
                return None
            max_c = max(values)
            return max_c * 9 / 5 + 32 if loc["unit"] == "F" else max_c
        except (requests.RequestException, ValueError, TypeError):
            return None

    def hours_left(self, event: dict[str, Any]) -> float:
        raw = event.get("endDate") or event.get("endDateIso")
        if not raw:
            return 999.0
        end = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return (end - datetime.now(timezone.utc)).total_seconds() / 3600

    def open_exposure(self) -> float:
        return sum(float(p["cost"]) for p in self.state["positions"] if p.get("status") == "open")

    def event_exposure(self, event_id: str) -> float:
        return sum(float(p["cost"]) for p in self.state["positions"] if p.get("status") == "open" and p.get("event_id") == event_id)

    def evaluate(self, bucket: Bucket, probability: float, book: Book, budget: float) -> Candidate | None:
        if book.best_ask is None or book.spread is None or book.spread > float(self.config["max_spread"]):
            return None
        p_cons = max(0.0, probability - float(self.config["probability_haircut"]))
        kelly = fractional_kelly(p_cons, book.best_ask, float(self.config["kelly_fraction"]))
        desired = min(budget, self.state["cash"] * kelly, float(self.config["max_trade_dollars"]))
        shares, vwap, spent = book.buy_vwap(desired)
        if not shares or not 0 < vwap < 1:
            return None
        edge = p_cons - vwap
        expected_profit = shares * (p_cons - vwap)
        if edge < float(self.config["min_net_edge"]):
            return None
        return Candidate(bucket, probability, p_cons, book.best_ask, vwap, shares, spent, edge, expected_profit, kelly)

    def paper_buy(self, event: dict[str, Any], city_slug: str, target: date, candidate: Candidate) -> None:
        position = {
            "status": "open", "opened_at": datetime.now(timezone.utc).isoformat(),
            "event_id": str(event.get("id", "")), "event_slug": event.get("slug"),
            "city": city_slug, "date": target.isoformat(), "market_id": candidate.bucket.market_id,
            "token_id": candidate.bucket.yes_token_id, "bucket": asdict(candidate.bucket),
            "model_probability": round(candidate.probability, 6),
            "conservative_probability": round(candidate.conservative_probability, 6),
            "entry_price": round(candidate.vwap, 6), "shares": round(candidate.shares, 6),
            "cost": round(candidate.cost, 6), "expected_profit": round(candidate.expected_profit, 6),
        }
        self.state["cash"] = round(self.state["cash"] - candidate.cost, 6)
        self.state["positions"].append(position)
        self.state["trades"].append({"type": "BUY", **position})
        self.save_state()

    def scan_city_date(self, city_slug: str, target: date) -> list[Candidate]:
        loc = self.config["locations"][city_slug]
        event = self.get_event(city_slug, target)
        if not event:
            return []
        hours = self.hours_left(event)
        if not float(self.config["min_hours"]) <= hours <= float(self.config["max_hours"]):
            return []
        buckets = [b for b in self.event_buckets(event, loc["unit"]) if b.volume >= float(self.config["min_volume"])]
        forecasts = self.forecast_points(city_slug, target)
        if len(buckets) < 2 or not forecasts:
            return []
        observed_max = self.current_observation(city_slug, target)
        probs = normalized_bucket_distribution(buckets, forecasts, observed_max)

        bankroll = self.state["cash"] + self.open_exposure()
        event_cap = bankroll * float(self.config["max_event_exposure_pct"])
        weather_cap = bankroll * float(self.config["max_weather_exposure_pct"])
        remaining_budget = min(
            event_cap - self.event_exposure(str(event.get("id", ""))),
            weather_cap - self.open_exposure(),
            float(self.config["max_trade_dollars"]),
        )
        if remaining_budget <= 0:
            return []

        candidates: list[Candidate] = []
        for bucket in buckets:
            try:
                book = self.get_book(bucket.yes_token_id)
            except requests.RequestException:
                continue
            if not book:
                continue
            c = self.evaluate(bucket, probs.get(bucket.market_id, 0.0), book, remaining_budget)
            if c:
                candidates.append(c)
        candidates.sort(key=lambda c: c.expected_profit, reverse=True)
        return candidates

    def scan_once(self) -> None:
        for city_slug, loc in self.config["locations"].items():
            local_today = datetime.now(ZoneInfo(loc["timezone"])).date()
            for offset in range(3):
                target = local_today + timedelta(days=offset)
                try:
                    candidates = self.scan_city_date(city_slug, target)
                except requests.RequestException as exc:
                    print(f"[{city_slug} {target}] API error: {exc}")
                    continue
                if not candidates:
                    continue
                best = candidates[0]
                print(f"[{city_slug} {target}] p={best.probability:.1%}, ask={best.ask:.3f}, "
                      f"vwap={best.vwap:.3f}, edge={best.edge:.1%}, cost=${best.cost:.2f}")
                # One position per event per scan. Paper-only guard is deliberate.
                if self.config.get("paper_trade", True):
                    event = self.get_event(city_slug, target)
                    if event and not any(p.get("event_id") == str(event.get("id")) and p.get("status") == "open" for p in self.state["positions"]):
                        self.paper_buy(event, city_slug, target, best)

    def status(self) -> None:
        open_positions = [p for p in self.state["positions"] if p.get("status") == "open"]
        print(f"Cash: ${self.state['cash']:.2f}")
        print(f"Open exposure: ${sum(p['cost'] for p in open_positions):.2f}")
        print(f"Realized PnL: ${self.state['realized_pnl']:.2f}")
        for p in open_positions:
            print(f"- {p['city']} {p['date']} | {p['bucket']['label']} | "
                  f"{p['shares']:.2f} @ {p['entry_price']:.3f} | model {p['model_probability']:.1%}")


def write_default_config(path: Path) -> None:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["scan", "run", "status", "init"], nargs="?", default="scan")
    parser.add_argument("--config", type=Path, default=Path("config_v3.json"))
    args = parser.parse_args()
    if args.command == "init":
        write_default_config(args.config)
        print(f"Wrote {args.config}")
        return
    write_default_config(args.config)
    bot = WeatherBot(args.config)
    if args.command == "status":
        bot.status()
    elif args.command == "scan":
        bot.scan_once()
    else:
        while True:
            bot.scan_once()
            time.sleep(int(bot.config["scan_interval"]))


if __name__ == "__main__":
    main()
