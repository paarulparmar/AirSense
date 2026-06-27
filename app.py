from __future__ import annotations

import math

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from pipeline.predict import run_prediction


st.set_page_config(page_title="AirSense", page_icon="AQI", layout="wide")


def secret_value(*names: str) -> str:
    for name in names:
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if value:
            return str(value)
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def cached_prediction(city: str, waqi_token: str, owm_api_key: str):
    return run_prediction(city, waqi_token, owm_api_key)


def risk_color(label: str) -> str:
    return {
        "Good": "#2ca25f",
        "Moderate": "#f0b429",
        "Unhealthy": "#f97316",
        "Hazardous": "#b91c1c",
    }.get(label, "#64748b")


def gauge(aqi: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi,
            number={"font": {"size": 44}},
            gauge={
                "axis": {
                    "range": [0, 300],
                    "tickmode": "array",
                    "tickvals": [0, 50, 100, 150, 200, 250, 300],
                    "ticktext": ["0", "50", "100", "150", "200", "250", "300"],
                },
                "bar": {"color": "#111827"},
                "steps": [
                    {"range": [0, 50], "color": "#b7e4c7"},
                    {"range": [50, 100], "color": "#fff3b0"},
                    {"range": [100, 200], "color": "#fdba74"},
                    {"range": [200, 300], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "value": min(aqi, 300),
                },
            },
            title={"text": "Current AQI"},
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=40, b=20),
    )

    return fig


def forecast_chart(forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["aqi"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#2563eb", width=3),
            name="Predicted AQI",
        )
    )
    fig.update_layout(
        height=340,
        xaxis_title="Time",
        yaxis_title="AQI",
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
    )
    return fig


def pollutant_card(name: str, value: float, unit: str, limit: float):
    pct = min(100, max(0, value / limit * 100 if limit else 0))
    st.metric(name, f"{value:.1f} {unit}")
    st.progress(int(pct))


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1180px;}
    .airsense-title {font-size: 2.4rem; font-weight: 750; margin: 0;}
    .subtle {color: #475569;}
    div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.85rem; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="airsense-title">AirSense</p>', unsafe_allow_html=True)
st.markdown('<p class="subtle">Live air quality, 72-hour ML forecast, and health guidance for any city.</p>', unsafe_allow_html=True)

waqi_token = secret_value("WAQI_TOKEN", "AQICN_TOKEN")
owm_key = secret_value("OPENWEATHER_API_KEY", "OWM_API_KEY")

with st.sidebar:
    st.header("City")
    city = st.text_input("Enter city name", value="Delhi")
    run = st.button("Analyze air quality", type="primary", use_container_width=True)
    st.caption("API responses are cached for 30 minutes.")
    if not waqi_token or not owm_key:
        st.info("Add WAQI_TOKEN and OPENWEATHER_API_KEY in Streamlit Secrets for live API data. Estimated fallback data is shown until then.")

if run or city:
    try:
        with st.spinner("Fetching live conditions and running forecast..."):
            result = cached_prediction(city.strip(), waqi_token, owm_key)
        live = result["live"]
        forecast = result["forecast"]
        label = result["risk_label"]
        color = risk_color(label)

        top = st.columns([1.1, 1.1, 0.9])
        with top[0]:
            st.plotly_chart(gauge(float(live["aqi"])), use_container_width=True)
        with top[1]:
            st.subheader(f"{live['city']}, {live.get('country', '')}")
            st.markdown(f"### <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            observed = live.get("observed_at", "")
            station = live.get("waqi_station", "")
            st.write(f"Station: {station}")
            st.write(f"Observed: {observed}")
            if live.get("is_estimated"):
                st.warning("Showing deterministic fallback estimates because one or more API calls were unavailable.")
        with top[2]:
            st.metric("Temperature", f"{float(live['temp']):.1f} °C")
            st.metric("Humidity", f"{float(live['humidity']):.0f}%")
            st.metric("Wind", f"{float(live['wind_speed']):.1f} m/s")

        st.divider()
        st.subheader("72-hour AQI forecast")
        st.plotly_chart(forecast_chart(forecast), use_container_width=True)

        st.subheader("Pollutants")
        pcols = st.columns(4)
        with pcols[0]:
            pollutant_card("PM2.5", float(live["pm25"]), "µg/m³", 35)
        with pcols[1]:
            pollutant_card("PM10", float(live["pm10"]), "µg/m³", 150)
        with pcols[2]:
            pollutant_card("NO2", float(live["no2"]), "ppb", 100)
        with pcols[3]:
            pollutant_card("O3", float(live["o3"]), "ppb", 70)

        st.subheader("Map")
        fmap = folium.Map(location=[live["lat"], live["lon"]], zoom_start=10, tiles="CartoDB positron")
        radius = max(8, min(34, math.sqrt(float(live["aqi"])) * 2))
        folium.CircleMarker(
            location=[live["lat"], live["lon"]],
            radius=radius,
            popup=f"{live['city']} AQI {float(live['aqi']):.0f}",
            color=color,
            fill=True,
            fill_opacity=0.75,
        ).add_to(fmap)
        st_folium(fmap, use_container_width=True, height=420)

        st.subheader("Health recommendations")
        cols = st.columns(2)
        for idx, recommendation in enumerate(result["recommendations"]):
            with cols[idx % 2]:
                st.info(recommendation)
    except Exception as exc:
        st.error(f"AirSense could not complete this lookup: {exc}")

