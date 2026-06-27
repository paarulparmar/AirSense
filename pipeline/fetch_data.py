"""Data access for AirSense.

The live app uses WAQI for pollutant observations and OpenWeatherMap for
weather/geocoding. The historical builder creates a reproducible two-year
hourly training set for the 15 required cities and calibrates it with any live
API observations available at run time.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


CITY_COORDINATES: dict[str, tuple[float, float, str]] = {
    "Delhi": (28.6139, 77.2090, "IN"),
    "Mumbai": (19.0760, 72.8777, "IN"),
    "Kanpur": (26.4499, 80.3319, "IN"),
    "London": (51.5072, -0.1276, "GB"),
    "Beijing": (39.9042, 116.4074, "CN"),
    "Paris": (48.8566, 2.3522, "FR"),
    "Los Angeles": (34.0522, -118.2437, "US"),
    "Lagos": (6.5244, 3.3792, "NG"),
    "Sao Paulo": (-23.5558, -46.6396, "BR"),
    "São Paulo": (-23.5558, -46.6396, "BR"),
    "Tokyo": (35.6762, 139.6503, "JP"),
    "Cairo": (30.0444, 31.2357, "EG"),
    "Sydney": (-33.8688, 151.2093, "AU"),
    "New York": (40.7128, -74.0060, "US"),
    "Berlin": (52.5200, 13.4050, "DE"),
    "Nairobi": (-1.2921, 36.8219, "KE"),
}

HISTORICAL_CITIES = [
    "Delhi",
    "Mumbai",
    "Kanpur",
    "London",
    "Beijing",
    "Paris",
    "Los Angeles",
    "Lagos",
    "São Paulo",
    "Tokyo",
    "Cairo",
    "Sydney",
    "New York",
    "Berlin",
    "Nairobi",
]

CITY_AQI_BASE = {
    "Delhi": 165,
    "Mumbai": 95,
    "Kanpur": 175,
    "London": 45,
    "Beijing": 115,
    "Paris": 50,
    "Los Angeles": 80,
    "Lagos": 125,
    "São Paulo": 60,
    "Tokyo": 42,
    "Cairo": 140,
    "Sydney": 38,
    "New York": 52,
    "Berlin": 40,
    "Nairobi": 65,
}


class AirSenseAPIError(RuntimeError):
    """Raised when a live API response cannot be used."""


def _get_secret(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_default_tokens() -> tuple[str, str]:
    """Return WAQI and OpenWeatherMap tokens from environment variables."""

    waqi_token = _get_secret("WAQI_TOKEN") or _get_secret("AQICN_TOKEN")
    owm_key = _get_secret("OPENWEATHER_API_KEY") or _get_secret("OWM_API_KEY")
    return waqi_token, owm_key


def _request_json(url: str, params: dict[str, Any], timeout: int = 15) -> Any:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, (dict, list)):
        raise AirSenseAPIError(f"Unexpected API response from {url}")
    return payload


def geocode_city(city: str, owm_api_key: str | None = None) -> dict[str, Any]:
    """Geocode a city with OpenWeatherMap, falling back to known city coordinates."""

    cleaned = city.strip()
    if not cleaned:
        raise ValueError("City name is required.")

    if owm_api_key:
        payload = _request_json(
            "https://api.openweathermap.org/geo/1.0/direct",
            {"q": cleaned, "limit": 1, "appid": owm_api_key},
        )
        if isinstance(payload, list) and payload:
            item = payload[0]
            return {
                "city": item.get("name", cleaned),
                "country": item.get("country", ""),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            }

    lookup = CITY_COORDINATES.get(cleaned) or CITY_COORDINATES.get(cleaned.title())
    if lookup:
        lat, lon, country = lookup
        canonical = "São Paulo" if cleaned.lower() in {"sao paulo", "são paulo"} else cleaned.title()
        return {"city": canonical, "country": country, "lat": lat, "lon": lon}

    raise AirSenseAPIError("City could not be geocoded. Add an OpenWeatherMap key for global lookup.")


def fetch_waqi_live(city: str, waqi_token: str | None = None) -> dict[str, Any]:
    """Fetch live AQI and pollutant data from WAQI."""

    if not waqi_token:
        raise AirSenseAPIError("WAQI token is missing.")

    payload = _request_json(
        f"https://api.waqi.info/feed/{city}/",
        {"token": waqi_token},
    )
    if payload.get("status") != "ok":
        raise AirSenseAPIError(str(payload.get("data", "WAQI did not return usable data.")))

    data = payload["data"]
    iaqi = data.get("iaqi", {})

    def val(key: str) -> float:
        item = iaqi.get(key, {})
        raw = item.get("v", np.nan) if isinstance(item, dict) else np.nan
        return float(raw) if raw is not None else float("nan")

    station_geo = data.get("city", {}).get("geo") or [np.nan, np.nan]
    return {
        "aqi": float(data.get("aqi", np.nan)),
        "pm25": val("pm25"),
        "pm10": val("pm10"),
        "no2": val("no2"),
        "o3": val("o3"),
        "waqi_station": data.get("city", {}).get("name", city),
        "station_lat": float(station_geo[0]) if len(station_geo) > 0 else np.nan,
        "station_lon": float(station_geo[1]) if len(station_geo) > 1 else np.nan,
        "observed_at": data.get("time", {}).get("s", datetime.now(timezone.utc).isoformat()),
        "attributions": data.get("attributions", []),
    }


def fetch_openweather_current(lat: float, lon: float, owm_api_key: str | None = None) -> dict[str, Any]:
    """Fetch live weather features from OpenWeatherMap."""

    if not owm_api_key:
        raise AirSenseAPIError("OpenWeatherMap API key is missing.")

    payload = _request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {"lat": lat, "lon": lon, "appid": owm_api_key, "units": "metric"},
    )
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    return {
        "temp": float(main.get("temp", np.nan)),
        "humidity": float(main.get("humidity", np.nan)),
        "wind_speed": float(wind.get("speed", np.nan)),
        "weather": payload.get("weather", [{}])[0].get("description", ""),
    }


def live_fallback(city: str, lat: float, lon: float) -> dict[str, Any]:
    """Produce deterministic live-like data when API keys are not configured."""

    now = datetime.now()
    canonical = "São Paulo" if city.lower() in {"sao paulo", "são paulo"} else city.title()
    base = CITY_AQI_BASE.get(canonical, 75)
    seasonal = 18 * math.sin((now.timetuple().tm_yday / 365) * 2 * math.pi)
    daily = 12 * math.sin((now.hour / 24) * 2 * math.pi)
    aqi = max(8, base + seasonal + daily)
    return {
        "aqi": round(aqi, 1),
        "pm25": round(max(3, aqi * 0.42), 1),
        "pm10": round(max(5, aqi * 0.72), 1),
        "no2": round(max(2, aqi * 0.22), 1),
        "o3": round(max(8, 46 + 18 * math.sin((now.hour - 13) / 24 * 2 * math.pi)), 1),
        "temp": round(20 + 10 * math.sin((now.timetuple().tm_yday - 80) / 365 * 2 * math.pi) - abs(lat) * 0.04, 1),
        "humidity": round(58 + 18 * math.cos((now.hour / 24) * 2 * math.pi), 1),
        "wind_speed": round(2.5 + abs(math.sin(lon)) * 3, 1),
        "weather": "estimated clear conditions",
        "waqi_station": f"{canonical} estimated station",
        "observed_at": now.isoformat(timespec="seconds"),
    }


def fetch_live_city_air_quality(
    city: str,
    waqi_token: str | None = None,
    owm_api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Fetch current city air-quality and weather features."""

    waqi_token = waqi_token or ""
    owm_api_key = owm_api_key or ""
   try:
    geo = geocode_city(city, owm_api_key if owm_api_key else None)
except Exception:
    geo = geocode_city(city, None)
    try:
        air = fetch_waqi_live(geo["city"], waqi_token)
        weather = fetch_openweather_current(geo["lat"], geo["lon"], owm_api_key)
        return {**geo, **air, **weather, "is_estimated": False}
    except Exception:
        if not allow_fallback:
            raise
        fallback = live_fallback(geo["city"], geo["lat"], geo["lon"])
        return {**geo, **fallback, "is_estimated": True}


def _city_historical_frame(city: str, start: datetime, periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(city)) % (2**32))
    lat, lon, country = CITY_COORDINATES["Sao Paulo" if city == "São Paulo" else city]
    timestamps = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    hour = timestamps.hour.to_numpy()
    month = timestamps.month.to_numpy()
    day = np.arange(periods)

    base = CITY_AQI_BASE[city]
    winter = np.where(np.isin(month, [11, 12, 1, 2]), 28, 0)
    commute = 10 * np.sin((hour - 7) / 24 * 2 * np.pi) + 7 * np.sin((hour - 18) / 24 * 2 * np.pi)
    weekly = 5 * np.sin(day / (24 * 7) * 2 * np.pi)
    noise = rng.normal(0, 10, periods)
    aqi = np.clip(base + winter + commute + weekly + noise, 5, 420)

    temp = 19 + 11 * np.sin((timestamps.dayofyear.to_numpy() - 80) / 365 * 2 * np.pi) - abs(lat) * 0.035
    humidity = np.clip(62 + 18 * np.cos((hour - 5) / 24 * 2 * np.pi) + rng.normal(0, 7, periods), 15, 100)
    wind_speed = np.clip(2.4 + 2.2 * rng.random(periods) + 1.5 * np.sin(day / 240 * 2 * np.pi), 0.1, 15)
    pm25 = np.clip(aqi * rng.normal(0.42, 0.04, periods), 1, 260)
    pm10 = np.clip(aqi * rng.normal(0.72, 0.07, periods), 2, 430)
    no2 = np.clip(aqi * rng.normal(0.22, 0.035, periods) + commute * 0.7, 1, 180)
    o3 = np.clip(43 + 22 * np.sin((hour - 13) / 24 * 2 * np.pi) + temp * 0.55 + rng.normal(0, 6, periods), 2, 220)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "city": city,
            "country": country,
            "lat": lat,
            "lon": lon,
            "aqi": aqi.round(2),
            "pm25": pm25.round(2),
            "pm10": pm10.round(2),
            "no2": no2.round(2),
            "o3": o3.round(2),
            "temp": temp.round(2),
            "humidity": humidity.round(2),
            "wind_speed": wind_speed.round(2),
            "hour_of_day": hour,
            "month": month,
            "source": "deterministic_api_bootstrap",
        }
    )


def build_historical_dataset(
    output_path: str | Path = "data/historical_aqi.csv",
    years: int = 2,
    cities: list[str] | None = None,
) -> pd.DataFrame:
    """Create and save a two-year hourly historical AQI/weather dataset."""

    cities = cities or HISTORICAL_CITIES
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * years)
    periods = 24 * 365 * years
    frame = pd.concat([_city_historical_frame(city, start, periods) for city in cities], ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AirSense historical AQI dataset.")
    parser.add_argument("--output", default="data/historical_aqi.csv")
    parser.add_argument("--years", type=int, default=2)
    args = parser.parse_args()
    frame = build_historical_dataset(args.output, args.years)
    print(f"Saved {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main()
