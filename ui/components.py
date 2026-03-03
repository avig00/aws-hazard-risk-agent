"""
Reusable Streamlit result display components.
Renders analytics tables, risk charts, source citations, and prediction cards.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Column display config ─────────────────────────────────────────────────────
# Maps Gold-layer column names → formatted Streamlit column configs.
_COL_CONFIG = {
    "county_name":              st.column_config.TextColumn("County"),
    "state":                    st.column_config.TextColumn("State"),
    "year":                     st.column_config.NumberColumn("Year", format="%d"),
    "avg_expected_loss":        st.column_config.NumberColumn("Avg Expected Loss ($)", format="$%.0f"),
    "avg_risk_score":           st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.3f"),
    "avg_vulnerability":        st.column_config.NumberColumn("Vulnerability", format="%.3f"),
    "avg_resilience":           st.column_config.NumberColumn("Resilience", format="%.3f"),
    "years_on_record":          st.column_config.NumberColumn("Years", format="%d"),
    "events_early_period":      st.column_config.NumberColumn("Events (Early Period)", format="%d"),
    "events_recent_period":     st.column_config.NumberColumn("Events (Recent Period)", format="%d"),
    "absolute_increase":        st.column_config.NumberColumn("Increase", format="%+.0f"),
    "pct_increase":             st.column_config.NumberColumn("% Change", format="%.1f%%"),
    "noaa_event_count":         st.column_config.NumberColumn("NOAA Events", format="%d"),
    "noaa_total_fatalities":    st.column_config.NumberColumn("Fatalities", format="%d"),
    "fema_declaration_count":   st.column_config.NumberColumn("FEMA Declarations", format="%d"),
    "fema_total_damage":        st.column_config.NumberColumn("FEMA Damage ($)", format="$%.0f"),
    "expected_loss":            st.column_config.NumberColumn("Expected Loss ($)", format="$%.2f"),
    "risk_score":               st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.4f"),
    "vulnerability":            st.column_config.NumberColumn("Vulnerability", format="%.4f"),
    "resilience":               st.column_config.NumberColumn("Resilience", format="%.4f"),
    "counties_affected":        st.column_config.NumberColumn("Counties Affected", format="%d"),
    "total_events":             st.column_config.NumberColumn("Total Events", format="%d"),
    "total_fatalities":         st.column_config.NumberColumn("Fatalities", format="%d"),
    "avg_property_damage":      st.column_config.NumberColumn("Avg Property Damage ($)", format="$%.0f"),
    "total_fema_declarations":  st.column_config.NumberColumn("FEMA Declarations", format="%d"),
    "avg_events_early":         st.column_config.NumberColumn("Avg Events (Early)", format="%.1f"),
    "avg_events_recent":        st.column_config.NumberColumn("Avg Events (Recent)", format="%.1f"),
}

# Columns that are internal IDs — hidden from the display table
_HIDDEN_COLS = {"county_fips"}

# Preferred metric for bar chart (first match wins)
_CHART_METRIC_PRIORITY = [
    "avg_expected_loss", "absolute_increase", "events_recent_period",
    "total_events", "avg_events_recent", "fema_declaration_count",
    "avg_risk_score",
]

# Human-readable chart axis labels
_CHART_LABELS = {
    "avg_expected_loss":    "Avg Expected Loss ($)",
    "absolute_increase":    "Increase in Events",
    "events_recent_period": "Events (Recent Period)",
    "total_events":         "Total Events",
    "avg_events_recent":    "Avg Events (Recent Period)",
    "fema_declaration_count": "FEMA Declarations",
    "avg_risk_score":       "Risk Score",
}


# ── Tool badges ───────────────────────────────────────────────────────────────

def render_tool_badges(tools_used: list, intent: str = "", reason: str = "") -> None:
    """
    Render color-coded pill badges for each tool used, plus optional intent label.
    query → blue  |  ask → purple  |  predict → teal
    """
    _BADGE = {
        "query":   ("#1E3A5F", "#60A5FA", "🔎 query"),
        "ask":     ("#3B1F5C", "#C084FC", "📚 ask"),
        "predict": ("#0F3D35", "#34D399", "🤖 predict"),
    }
    parts = []
    for tool in tools_used:
        bg, fg, label = _BADGE.get(tool, ("#1F2937", "#9CA3AF", f"⚙️ {tool}"))
        parts.append(
            f'<span style="background:{bg};color:{fg};padding:3px 10px;'
            f'border-radius:12px;font-size:0.72rem;font-weight:600;'
            f'margin-right:6px;letter-spacing:0.03em">{label}</span>'
        )
    if intent and intent not in ("—", ""):
        parts.append(
            f'<span style="color:#4B5563;font-size:0.72rem">'
            f'intent: <code style="font-size:0.70rem;color:#6B7280">{intent}</code></span>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


# ── Analytics table ───────────────────────────────────────────────────────────

def render_analytics_table(results: list, title: str = "Results") -> None:
    """
    Display Athena query results as a formatted, interactive dataframe.
    - Numeric columns are formatted with $ / % symbols
    - FIPS codes are hidden
    - Auto-selects the most meaningful metric for the bar chart
    """
    if not results:
        st.info("No results returned.")
        return

    df = pd.DataFrame(results)

    # Convert numeric-looking columns from string (Athena returns everything as str)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Hide internal ID columns
    display_cols = [c for c in df.columns if c not in _HIDDEN_COLS]
    display_df = df[display_cols]

    col_config = {c: cfg for c, cfg in _COL_CONFIG.items() if c in display_df.columns}

    st.markdown(
        f'<p style="color:#6B7280;font-size:0.8rem;margin-bottom:4px">'
        f'{len(df)} rows returned</p>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        display_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
    )

    # Chart: pick the most meaningful numeric metric
    numeric_cols = display_df.select_dtypes(include="number").columns.tolist()
    label_col = next(
        (c for c in ["county_name", "state", "year"] if c in display_df.columns), None
    )
    metric_col = next(
        (c for c in _CHART_METRIC_PRIORITY if c in numeric_cols),
        next((c for c in numeric_cols if c not in {"year"}), None),
    )

    if not metric_col or not label_col or label_col == "year":
        return
    if len(display_df) > 50:
        return

    # Blue gradient: dark navy (low) → sky blue (high) — matches dark theme
    color_scale = [[0, "#1E2535"], [0.5, "#1D4ED8"], [1.0, "#4F9CF9"]]

    y_label = _CHART_LABELS.get(metric_col, metric_col.replace("_", " ").title())

    fig = px.bar(
        display_df.head(20),
        x=label_col,
        y=metric_col,
        color=metric_col,
        color_continuous_scale=color_scale,
        labels={metric_col: y_label, label_col: ""},
        height=320,
    )
    fig.update_layout(
        xaxis_tickangle=-40,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=12, b=80),
        font=dict(size=11, color="#E6EDF3"),
        xaxis=dict(gridcolor="#1E2535"),
        yaxis=dict(gridcolor="#1E2535"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Trend chart ───────────────────────────────────────────────────────────────

def render_trend_chart(results: list, x_col: str = "year", y_col: str = "total_events") -> None:
    """Line chart for year-over-year trend data."""
    if not results:
        return

    df = pd.DataFrame(results)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    if x_col not in df.columns:
        return

    # Auto-select y if default not present
    if y_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_year = [c for c in numeric_cols if c != x_col]
        if not non_year:
            return
        y_col = non_year[0]

    y_label = _CHART_LABELS.get(y_col, y_col.replace("_", " ").title())

    fig = px.line(
        df, x=x_col, y=y_col,
        labels={y_col: y_label, x_col: "Year"},
        markers=True,
        height=300,
    )
    fig.update_traces(line_color="#4F9CF9", marker_color="#4F9CF9")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=12, b=40),
        font=dict(size=11, color="#E6EDF3"),
        xaxis=dict(gridcolor="#1E2535"),
        yaxis=dict(gridcolor="#1E2535"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Risk map ──────────────────────────────────────────────────────────────────

def render_risk_map(results: list) -> None:
    """
    County risk map using pydeck.
    Requires latitude/longitude in results — skips gracefully if not present.
    """
    df = pd.DataFrame(results)
    if "lat" not in df.columns or "lon" not in df.columns:
        return

    try:
        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_color="[200, 30, 0, 160]",
            get_radius="avg_expected_loss / 100",
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=37.5, longitude=-96, zoom=3.5)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
    except ImportError:
        st.info("Install pydeck for map visualization: pip install pydeck")


# ── Prediction card ───────────────────────────────────────────────────────────

def render_prediction_card(prediction: dict) -> None:
    """Display county ML risk prediction with a color-coded tier badge."""
    county = prediction.get("county_name") or prediction.get("county_fips") or "County"
    risk_tier = prediction.get("risk_tier", "—")
    probs = prediction.get("probabilities", {})

    _TIER = {
        "HIGH":   ("#7F1D1D", "#FCA5A5", "#EF4444", "🔴"),
        "MEDIUM": ("#78350F", "#FCD34D", "#F59E0B", "🟡"),
        "LOW":    ("#14532D", "#86EFAC", "#22C55E", "🟢"),
    }
    bg, text_color, accent, icon = _TIER.get(risk_tier, ("#1F2937", "#9CA3AF", "#6B7280", "⚪"))

    st.markdown(
        f"""
        <div style="background:{bg};border-left:4px solid {accent};
                    border-radius:6px;padding:12px 16px;margin:8px 0">
          <div style="color:#9CA3AF;font-size:0.75rem;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:2px">ML Risk Prediction</div>
          <div style="color:#F9FAFB;font-size:1rem;font-weight:600">{county}</div>
          <div style="color:{text_color};font-size:1.4rem;font-weight:700;margin-top:4px">
            {icon} {risk_tier}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probs:
        cols = st.columns(3)
        tier_order = ["LOW", "MEDIUM", "HIGH"]
        for col, tier in zip(cols, tier_order):
            p = probs.get(tier, 0.0)
            _, tc, ac, ic = _TIER.get(tier, ("#1F2937", "#9CA3AF", "#6B7280", "⚪"))
            with col:
                st.markdown(
                    f'<div style="text-align:center;color:{tc};font-size:0.85rem">'
                    f'{ic} {tier}<br>'
                    f'<span style="font-size:1.1rem;font-weight:700">{p:.0%}</span></div>',
                    unsafe_allow_html=True,
                )


# ── Citations ─────────────────────────────────────────────────────────────────

def render_citations(sources: list) -> None:
    """Display RAG source citations in a compact expandable section."""
    if not sources:
        return

    with st.expander(f"📚 Sources — {len(sources)} document{'s' if len(sources) != 1 else ''}"):
        for i, src in enumerate(sources, 1):
            doc = src.get("source", "Unknown document")
            hazard = src.get("hazard_type", "")
            score = src.get("score", 0.0)
            relevance_bar = "█" * round(score * 10)
            hazard_str = f" · {hazard}" if hazard and hazard != "general" else ""
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid #1E2535">'
                f'<span style="color:#4F9CF9;font-size:0.8rem;font-weight:600">{i}.</span> '
                f'<code style="font-size:0.78rem">{doc}</code>'
                f'<span style="color:#6B7280;font-size:0.75rem">{hazard_str}</span><br>'
                f'<span style="color:#374151;font-size:0.7rem" title="Relevance: {score:.3f}">'
                f'{relevance_bar}</span>'
                f'<span style="color:#6B7280;font-size:0.7rem"> {score:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── SQL expander ──────────────────────────────────────────────────────────────

def render_sql_expander(sql: str, scan_bytes: int = None) -> None:
    """Show the compiled SQL in a collapsible expander."""
    with st.expander("🔍 View executed SQL"):
        st.code(sql, language="sql")
        if scan_bytes:
            mb = scan_bytes / (1024 * 1024)
            st.caption(f"Data scanned: {mb:.2f} MB")


# ── Error / warning states ────────────────────────────────────────────────────

def render_error(message: str) -> None:
    st.error(f"**Error:** {message}")


def render_no_data(message: str = None) -> None:
    """Styled warning for queries that return no rows."""
    msg = message or "No matching records found in the Gold-layer dataset. Try broadening the year range or hazard type."
    st.warning(msg, icon="⚠️")
