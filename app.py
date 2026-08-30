"""
NetSage AI — Streamlit Dashboard
app.py
"""
import os, sys, json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.rule_checker import run_all_checks, findings_to_dicts
from src.ai_agent import diagnose, get_demo_diagnosis, DiagnosisResult, ErrorResult

# ── file paths ────────────────────────────────────────────────────────────────
CASES_CSV   = ROOT / "data" / "cases.csv"
REVIEWS_CSV = ROOT / "data" / "reviews.csv"

REVIEWS_COLS = [
    "timestamp", "case_id", "symptom", "concept",
    "ai_root_cause", "ai_confidence", "human_action",
    "human_correction", "reviewer_note",
]

SEVERITY_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f97316",
    "Medium":   "#eab308",
    "Low":      "#22c55e",
    "Info":     "#94a3b8",
}

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_cases() -> pd.DataFrame:
    return pd.read_csv(CASES_CSV, dtype=str).fillna("")

def load_reviews() -> pd.DataFrame:
    if REVIEWS_CSV.exists():
        return pd.read_csv(REVIEWS_CSV, dtype=str).fillna("")
    return pd.DataFrame(columns=REVIEWS_COLS)

def save_review(row: dict):
    df = load_reviews()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    REVIEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REVIEWS_CSV, index=False)

def severity_badge(sev: str) -> str:
    color = SEVERITY_COLORS.get(sev, "#94a3b8")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">{sev}</span>'

def confidence_badge(conf: str) -> str:
    colors = {"High": "#22c55e", "Medium": "#eab308", "Low": "#ef4444"}
    color = colors.get(conf, "#94a3b8")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">{conf}</span>'

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetSage AI — Network Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* dark background */
.stApp { background: #0f1117; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #141923 !important; border-right: 1px solid #1e2940; }

/* metric cards */
.metric-card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; }

/* section headers */
.section-header {
    font-size: 1.05rem; font-weight: 600;
    color: #38bdf8; letter-spacing: .04em;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px; margin-bottom: 14px;
}

/* finding card */
.finding-card {
    background: #1a2235;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
}
.finding-card.Critical { border-left-color: #ef4444; }
.finding-card.High     { border-left-color: #f97316; }
.finding-card.Medium   { border-left-color: #eab308; }
.finding-card.Low      { border-left-color: #22c55e; }

/* show output pre */
.cli-block {
    background: #0d1117;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #a3e635;
    white-space: pre-wrap;
    overflow-x: auto;
    max-height: 340px;
    overflow-y: auto;
}

/* AI result block */
.ai-card {
    background: linear-gradient(135deg,#0c1a2e,#0f1f35);
    border: 1px solid #1e4d7a;
    border-radius: 12px;
    padding: 20px;
}
.fix-step {
    background: #0d1117;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #a3e635;
    margin: 4px 0;
}

/* human review panel */
.review-panel {
    background: #1a2235;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px;
}

/* plotly charts transparent bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* top bar logo */
.logo-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0 16px 0;
}
.logo-text { font-size: 1.5rem; font-weight: 700; color: #38bdf8; }
.logo-sub  { font-size: 0.8rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
cases_df = load_cases()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
        <span style="font-size:2rem">🔬</span>
        <div>
            <div class="logo-text">NetSage AI</div>
            <div class="logo-sub">Network Fault Diagnostics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Select Lab Case")

    case_options = {
        f"{row['case_id']} — {row['concept']}": i
        for i, row in cases_df.iterrows()
    }
    selected_label = st.selectbox(
        "Lab Case",
        list(case_options.keys()),
        label_visibility="collapsed",
    )
    selected_idx = case_options[selected_label]
    case = cases_df.iloc[selected_idx].to_dict()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    use_demo = st.toggle("Demo Mode (no API key needed)", value=True)
    if not use_demo:
        api_key_input = st.text_input("OpenAI API Key", type="password",
                                      value=os.getenv("OPENAI_API_KEY",""),
                                      help="Enter your key or add it to .env")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

    st.markdown("---")
    st.caption("Human-in-the-Loop AI · Responsible AI Log enabled")
    reviews_df = load_reviews()
    total_rev = len(reviews_df)
    corrections = len(reviews_df[reviews_df["human_action"].isin(["Edited","Rejected"])]) if total_rev else 0
    st.metric("Total Reviews", total_rev)
    st.metric("Human Corrections", corrections)

# ── main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:6px 0 20px 0">
    <h1 style="font-size:1.9rem;font-weight:700;color:#f8fafc;margin:0">
        🔬 NetSage AI — Network Fault Diagnostics
    </h1>
    <p style="color:#64748b;margin:4px 0 0 0;font-size:0.9rem">
        Human-in-the-loop AI assistant for Cisco Packet Tracer labs
    </p>
</div>
""", unsafe_allow_html=True)

tab_diag, tab_dashboard = st.tabs(["🩺 Diagnose", "📊 Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_diag:

    # ── case detail ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">📁 Case Details</div>', unsafe_allow_html=True)

        sev_color = SEVERITY_COLORS.get(case.get("severity",""), "#94a3b8")
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:1.1rem;font-weight:600;color:#f1f5f9">{case['case_id']}</span>
                <span style="background:{sev_color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">{case.get('severity','')}</span>
            </div>
            <div style="color:#cbd5e1;font-size:0.9rem;margin-bottom:6px"><b>Concept:</b> {case.get('concept','')}</div>
            <div style="color:#94a3b8;font-size:0.8rem"><b>OSI Layer:</b> {case.get('osi_layer','')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🚨 Symptom**")
        st.info(case.get("symptom", ""))

        st.markdown("**🗺️ Topology**")
        st.markdown(f'<div style="background:#1a2235;border-radius:8px;padding:12px;font-size:0.88rem;color:#cbd5e1">{case.get("topology_note","")}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">💻 Show Command Outputs</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cli-block">{case.get("show_outputs","")}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── rule checker ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Automatic Rule Checker</div>', unsafe_allow_html=True)

    findings = run_all_checks(case.get("show_outputs", ""))

    if findings:
        c1, c2, c3 = st.columns(3)
        critical_n = sum(1 for f in findings if f.severity == "Critical")
        high_n     = sum(1 for f in findings if f.severity == "High")
        other_n    = len(findings) - critical_n - high_n
        c1.metric("🔴 Critical", critical_n)
        c2.metric("🟠 High",     high_n)
        c3.metric("🟡 Other",    other_n)

        for f in findings:
            sev = f.severity
            st.markdown(f"""
            <div class="finding-card {sev}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                    <span style="font-weight:600;color:#f1f5f9">{f.rule_id} — {f.message}</span>
                    {severity_badge(sev)}
                </div>
                <div style="color:#94a3b8;font-size:0.82rem;font-family:monospace">{f.matched_text}</div>
                <div style="color:#38bdf8;font-size:0.82rem;margin-top:6px">💡 {f.recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No deterministic rule violations found in the show output.")

    st.markdown("---")

    # ── AI diagnosis ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 AI Agent Diagnosis</div>', unsafe_allow_html=True)

    if "ai_result" not in st.session_state:
        st.session_state.ai_result = None
    if "current_case" not in st.session_state:
        st.session_state.current_case = None

    # Reset AI result when case changes
    if st.session_state.current_case != case["case_id"]:
        st.session_state.ai_result = None
        st.session_state.current_case = case["case_id"]

    col_btn1, col_btn2 = st.columns([1,5])
    with col_btn1:
        run_ai = st.button("▶ Run AI Diagnosis", type="primary", use_container_width=True)

    if run_ai:
        with st.spinner("🧠 AI is analyzing the case..."):
            if use_demo:
                result = get_demo_diagnosis(case)
            else:
                result = diagnose(case)
        st.session_state.ai_result = result

    ai_result = st.session_state.ai_result

    if ai_result is not None:
        if isinstance(ai_result, ErrorResult):
            st.error(f"**AI Error:** {ai_result.error}")
            if ai_result.raw_response:
                with st.expander("Raw API Response"):
                    st.code(ai_result.raw_response)
        else:
            # Show structured result
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)

            rc1, rc2 = st.columns([2,1])
            with rc1:
                st.markdown("#### 🔎 Root Cause")
                st.markdown(f'<div style="font-size:1rem;color:#f1f5f9;font-weight:500">{ai_result.root_cause}</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown("#### Confidence")
                st.markdown(confidence_badge(ai_result.confidence), unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8;font-size:0.8rem;margin-top:4px'>OSI: {ai_result.osi_layer}</div>", unsafe_allow_html=True)

            st.markdown("---")
            ev_col, fix_col = st.columns(2, gap="large")

            with ev_col:
                st.markdown("**📌 Evidence from Show Output**")
                for e in ai_result.evidence:
                    st.markdown(f'<div style="background:#0d1117;border-left:3px solid #38bdf8;padding:6px 10px;margin:4px 0;font-family:monospace;font-size:0.8rem;color:#a3e635;border-radius:4px">{e}</div>', unsafe_allow_html=True)

                st.markdown("**🔭 Next Recommended Command**")
                st.markdown(f'<div class="fix-step">$ {ai_result.next_command}</div>', unsafe_allow_html=True)

            with fix_col:
                st.markdown("**🛠️ Fix Steps**")
                for i, step in enumerate(ai_result.fix_steps, 1):
                    st.markdown(f'<div class="fix-step"><span style="color:#64748b">{i}.</span> {step}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**💡 Beginner Explanation**")
            st.markdown(f'<div style="background:#0f1f35;border-radius:8px;padding:14px;color:#cbd5e1;font-size:0.9rem;line-height:1.6">{ai_result.beginner_explanation}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── Human Review Panel ────────────────────────────────────────────
            st.markdown('<div class="section-header">👤 Human Review</div>', unsafe_allow_html=True)
            st.markdown('<div class="review-panel">', unsafe_allow_html=True)
            st.markdown("**Review the AI diagnosis above and record your decision:**")

            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                accept_btn = st.button("✅ Accept", use_container_width=True, key="btn_accept")
            with hcol2:
                edit_btn   = st.button("✏️ Edit / Correct", use_container_width=True, key="btn_edit")
            with hcol3:
                reject_btn = st.button("❌ Reject", use_container_width=True, key="btn_reject")

            human_correction = ""
            reviewer_note = ""
            action = None

            if accept_btn:
                action = "Accepted"
                reviewer_note = st.text_area("Optional note:", key="note_accept", height=80)
            elif edit_btn:
                action = "Edited"
                human_correction = st.text_area(
                    "Your corrected diagnosis:", key="correction_edit", height=80,
                    placeholder="Describe the actual fault and correct fix..."
                )
                reviewer_note = st.text_area("Reason for correction:", key="note_edit", height=60)
            elif reject_btn:
                action = "Rejected"
                human_correction = st.text_area(
                    "What is the actual issue?", key="correction_reject", height=80,
                    placeholder="The AI was wrong because..."
                )
                reviewer_note = st.text_area("Explanation:", key="note_reject", height=60)

            if action:
                col_save, _ = st.columns([1,4])
                with col_save:
                    if st.button("💾 Save Review", type="primary", key="save_review"):
                        row = {
                            "timestamp":       datetime.now().isoformat(timespec="seconds"),
                            "case_id":         case["case_id"],
                            "symptom":         case.get("symptom","")[:80],
                            "concept":         case.get("concept",""),
                            "ai_root_cause":   ai_result.root_cause[:100],
                            "ai_confidence":   ai_result.confidence,
                            "human_action":    action,
                            "human_correction": human_correction[:200],
                            "reviewer_note":   reviewer_note[:200],
                        }
                        save_review(row)
                        st.cache_data.clear()
                        if action == "Accepted":
                            st.success("✅ Review saved — AI diagnosis accepted.")
                        else:
                            st.warning(f"⚠️ Review saved — AI diagnosis {action}.")

            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Oversight & Metrics Dashboard")
    reviews_df = load_reviews()

    CHART_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,17,23,0.5)",
        font=dict(color="#e2e8f0", family="Inter"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    if reviews_df.empty:
        st.info("No reviews logged yet. Run a diagnosis and submit a Human Review to see data here.")
        st.markdown("---")
        st.markdown("#### 📋 Case Dataset Overview")
        fig_cases = px.bar(
            cases_df["concept"].value_counts().reset_index(),
            x="concept", y="count",
            title="Lab Cases by Concept",
            color="concept",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_cases.update_layout(**CHART_LAYOUT)
        fig_cases.update_xaxes(title="", tickangle=-30)
        fig_cases.update_yaxes(title="Count")
        st.plotly_chart(fig_cases, use_container_width=True)

    else:
        # ── top metrics ───────────────────────────────────────────────────────
        total   = len(reviews_df)
        accepted = (reviews_df["human_action"] == "Accepted").sum()
        edited   = (reviews_df["human_action"] == "Edited").sum()
        rejected = (reviews_df["human_action"] == "Rejected").sum()
        accuracy = round(accepted / total * 100, 1) if total else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, value, delta_color in [
            (m1, "Total Reviews",  total,    "normal"),
            (m2, "AI Accepted",    accepted, "normal"),
            (m3, "Human Edited",   edited,   "inverse"),
            (m4, "AI Rejected",    rejected, "inverse"),
            (m5, "AI Accuracy %",  f"{accuracy}%", "normal"),
        ]:
            col.metric(label, value)

        st.markdown("---")

        # ── charts row 1 ──────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            action_counts = reviews_df["human_action"].value_counts().reset_index()
            action_counts.columns = ["Action","Count"]
            fig_pie = px.pie(
                action_counts, names="Action", values="Count",
                title="Human Decision Distribution",
                color="Action",
                color_discrete_map={
                    "Accepted": "#22c55e",
                    "Edited":   "#eab308",
                    "Rejected": "#ef4444",
                },
                hole=0.45,
            )
            fig_pie.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            conf_action = reviews_df.groupby(["ai_confidence","human_action"]).size().reset_index(name="count")
            fig_conf = px.bar(
                conf_action,
                x="ai_confidence", y="count", color="human_action",
                barmode="group",
                title="Confidence Level vs Human Decision",
                color_discrete_map={
                    "Accepted": "#22c55e",
                    "Edited":   "#eab308",
                    "Rejected": "#ef4444",
                },
                category_orders={"ai_confidence": ["High","Medium","Low"]},
            )
            fig_conf.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig_conf, use_container_width=True)

        # ── charts row 2 ──────────────────────────────────────────────────────
        ch3, ch4 = st.columns(2, gap="large")

        with ch3:
            concept_counts = reviews_df["concept"].value_counts().reset_index()
            concept_counts.columns = ["Concept","Reviews"]
            fig_conc = px.bar(
                concept_counts, x="Reviews", y="Concept",
                orientation="h",
                title="Reviews by Concept",
                color="Reviews",
                color_continuous_scale="Blues",
            )
            fig_conc.update_layout(**CHART_LAYOUT)
            fig_conc.update_coloraxes(showscale=False)
            st.plotly_chart(fig_conc, use_container_width=True)

        with ch4:
            if "timestamp" in reviews_df.columns and reviews_df["timestamp"].notna().any():
                reviews_df["date"] = pd.to_datetime(reviews_df["timestamp"], errors="coerce").dt.date
                timeline = reviews_df.groupby(["date","human_action"]).size().reset_index(name="count")
                fig_time = px.line(
                    timeline, x="date", y="count", color="human_action",
                    title="Review Activity Over Time",
                    markers=True,
                    color_discrete_map={
                        "Accepted": "#22c55e",
                        "Edited":   "#eab308",
                        "Rejected": "#ef4444",
                    },
                )
                fig_time.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Timeline chart available after multiple review sessions.")

        st.markdown("---")

        # ── responsible AI log ────────────────────────────────────────────────
        st.markdown("#### 📜 Responsible AI Log — Human Corrections")
        corrections_df = reviews_df[reviews_df["human_action"].isin(["Edited","Rejected"])].copy()
        if corrections_df.empty:
            st.info("No corrections logged yet. When you edit or reject an AI diagnosis, entries appear here.")
        else:
            st.markdown(f"**{len(corrections_df)} correction(s) logged** *(target: 5+)*")
            display_cols = ["timestamp","case_id","concept","ai_root_cause",
                            "ai_confidence","human_action","human_correction","reviewer_note"]
            st.dataframe(
                corrections_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=320,
            )
            csv_bytes = corrections_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Corrections CSV",
                csv_bytes,
                "responsible_ai_log.csv",
                "text/csv",
            )

        st.markdown("---")
        st.markdown("#### 📄 Full Review Log")
        st.dataframe(reviews_df.reset_index(drop=True), use_container_width=True, height=260)
        st.download_button(
            "⬇️ Download Full Reviews CSV",
            reviews_df.to_csv(index=False).encode(),
            "reviews.csv", "text/csv",
        )
