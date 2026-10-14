import os
import threading
import time

import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from exam_monitor import (
    BASE_DIR,
    LOG_FILE,
    SESSION_EVENTS_FILE,
    SESSION_SUMMARY_FILE,
    ExamMonitorProcessor,
)

st.set_page_config(
    page_title="AI Exam Monitor",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1e3a8a);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .4rem 0; color: white;}
    .hero p {margin: 0; opacity: .9;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>AI Exam Monitor</h1>
  <p>Computer vision monitoring · Live review indicators · Local session reports</p>
</div>
""", unsafe_allow_html=True)

# Only one monitoring session is intended per local app instance.
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "processor" not in st.session_state:
    st.session_state.processor = None
if "last_session_summary" not in st.session_state:
    st.session_state.last_session_summary = None

def read_csv_safely(path):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except (pd.errors.EmptyDataError, PermissionError, OSError):
        pass
    return pd.DataFrame()

def make_processor():
    # The processor is retrieved from the WebRTC context after it starts.
    # Avoid accessing Streamlit session state from the video-processing thread.
    return ExamMonitorProcessor()

with st.sidebar:
    st.header("Controls")
    st.caption("Your webcam and AI processing run locally on this computer.")
    st.info("Allow camera access when your browser asks for permission.")
    st.markdown("**Local output files**")
    st.code("exam_events.csv\nexam_session_events.csv\nexam_session_summary.csv")

left, right = st.columns([1.6, 1])

with left:
    st.subheader("Live monitoring")
    if not st.session_state.monitoring:
        st.write("Start a session to open the live webcam feed.")
        if st.button("▶ Start Monitoring", type="primary", use_container_width=True):
            st.session_state.monitoring = True
            st.session_state.processor = None
            st.rerun()
    else:
        st.success("Monitoring is active. Look at the screen during the initial calibration.")
        ctx = webrtc_streamer(
            key="exam-monitor-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=make_processor,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )

        # When the video processor is available, show live status/metrics.
        processor = ctx.video_processor
        if processor is not None:
            st.session_state.processor = processor

        if processor is not None and not processor.stopped:
            metrics = processor.last_metrics
            if metrics:
                st.markdown("#### Current frame")
                c1, c2, c3 = st.columns(3)
                c1.metric("People", metrics.get("People", 0))
                c2.metric("Faces", metrics.get("Faces", 0))
                c3.metric("Head status", metrics.get("Head status", "Starting"))
                c4, c5 = st.columns(2)
                c4.metric("Phone detected", "Yes" if metrics.get("Phone detected") else "No")
                c5.metric("Book detected", "Yes" if metrics.get("Book detected") else "No")

        if st.button("■ Stop Session & Save Report", type="secondary", use_container_width=True):
            processor = st.session_state.processor
            if processor is not None:
                processor.stop()
                st.session_state.last_session_summary = processor.get_live_summary()
            st.session_state.monitoring = False
            st.session_state.processor = None
            st.rerun()

with right:
    st.subheader("Session overview")
    summaries = read_csv_safely(SESSION_SUMMARY_FILE)
    events = read_csv_safely(SESSION_EVENTS_FILE)
    alerts = read_csv_safely(LOG_FILE)

    total_sessions = len(summaries)
    total_alerts = len(alerts)
    total_event_intervals = len(events)

    m1, m2 = st.columns(2)
    m1.metric("Saved sessions", total_sessions)
    m2.metric("Alert log rows", total_alerts)
    m3, m4 = st.columns(2)
    m3.metric("Event intervals", total_event_intervals)

    if not summaries.empty and "session_duration_seconds" in summaries.columns:
        total_minutes = pd.to_numeric(
            summaries["session_duration_seconds"], errors="coerce"
        ).fillna(0).sum() / 60
    else:
        total_minutes = 0
    m4.metric("Total monitored", f"{total_minutes:.1f} min")

if st.session_state.last_session_summary:
    st.success("The last session has been saved.")
    summary = st.session_state.last_session_summary
    st.write(f"**Session ID:** `{summary.get('session_id', '—')}`")
    st.write(
        f"**Duration:** {summary.get('session_duration_seconds', 0)} seconds"
    )

st.divider()
st.subheader("Analytics")

summaries = read_csv_safely(SESSION_SUMMARY_FILE)
events = read_csv_safely(SESSION_EVENTS_FILE)
alerts = read_csv_safely(LOG_FILE)

tab1, tab2, tab3 = st.tabs(["Session history", "Event durations", "Alert log"])

with tab1:
    if summaries.empty:
        st.info("No completed sessions yet. Start and stop a session to populate this table.")
    else:
        st.dataframe(summaries.iloc[::-1], use_container_width=True, hide_index=True)
        if "session_duration_seconds" in summaries.columns:
            chart_data = summaries.copy()
            chart_data["session_duration_seconds"] = pd.to_numeric(
                chart_data["session_duration_seconds"], errors="coerce"
            )
            if "session_id" in chart_data.columns:
                st.markdown("**Session duration (seconds)**")
                st.bar_chart(
                    chart_data.set_index("session_id")[["session_duration_seconds"]]
                )
        st.download_button(
            "Download session summary CSV",
            data=summaries.to_csv(index=False).encode("utf-8"),
            file_name="exam_session_summary.csv",
            mime="text/csv",
        )

with tab2:
    if events.empty:
        st.info("No event intervals recorded yet.")
    else:
        st.dataframe(events.iloc[::-1], use_container_width=True, hide_index=True)
        if "duration_seconds" in events.columns and "event" in events.columns:
            duration_data = events.copy()
            duration_data["duration_seconds"] = pd.to_numeric(
                duration_data["duration_seconds"], errors="coerce"
            ).fillna(0)
            by_event = duration_data.groupby("event")["duration_seconds"].sum().sort_values()
            st.markdown("**Total recorded seconds by event type**")
            st.bar_chart(by_event)
        st.download_button(
            "Download event durations CSV",
            data=events.to_csv(index=False).encode("utf-8"),
            file_name="exam_session_events.csv",
            mime="text/csv",
        )

with tab3:
    if alerts.empty:
        st.info("No alerts logged yet.")
    else:
        st.dataframe(alerts.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download alert log CSV",
            data=alerts.to_csv(index=False).encode("utf-8"),
            file_name="exam_events.csv",
            mime="text/csv",
        )

st.caption(
    "Review indicators can be inaccurate. A detected phone, book, head turn, or additional person "
    "is not proof of misconduct. Use human review and obtain appropriate consent before monitoring."
)
