"""
Reusable Streamlit result display components.
Renders analytics tables, risk charts, source citations, and prediction cards.
"""
import pandas as pd
import plotly.express as px
import streamlit as st


def render_analytics_table(results: list, title: str = "Results") -> None:
    """Display Athena query results as an interactive dataframe."""
    if not results:
        st.info("No results returned.")
        return

    df = pd.DataFrame(results)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

    # Auto-detect if a bar chart makes sense (numeric + county name)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    label_col = next(
        (c for c in ["county_name", "state", "year"] if c in df.columns), None
    )

    if numeric_cols and label_col and len(df) <= 50:
        metric_col = numeric_cols[0]
        fig = px.bar(
            df.head(20),
            x=label_col,
            y=metric_col,
            title=f"{metric_col} by {label_col}",
            color=metric_col,
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_trend_chart(results: list, x_col: str = "year", y_col: str = "total_events") -> None:
    """Line chart for year-over-year trend data."""
    if not results:
        return

    df = pd.DataFrame(results)
    if x_col not in df.columns or y_col not in df.columns:
        return

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=f"{y_col.replace('_', ' ').title()} Over Time",
        markers=True,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


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


def render_prediction_card(prediction: dict) -> None:
    """Display a county ML risk prediction as a metric card."""
    county = prediction.get("county_name") or prediction.get("county_fips") or "County"
    risk_tier = prediction.get("risk_tier", "—")

    color_map = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    icon = color_map.get(risk_tier, "⚪")

    st.subheader("ML Risk Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="County", value=county)
    with col2:
        st.metric(label="Predicted Risk Tier", value=f"{icon} {risk_tier}")


def render_citations(sources: list) -> None:
    """Display RAG source citations in an expandable section."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} documents)"):
        for i, src in enumerate(sources, 1):
            doc = src.get("source", "Unknown document")
            hazard = src.get("hazard_type", "")
            score = src.get("score", 0.0)
            st.markdown(
                f"**{i}.** `{doc}`"
                + (f" | Hazard: *{hazard}*" if hazard and hazard != "general" else "")
                + f" | Relevance: `{score:.3f}`"
            )


def render_sql_expander(sql: str, scan_bytes: int = None) -> None:
    """Show the compiled SQL in a collapsible expander."""
    with st.expander("🔍 View executed SQL"):
        st.code(sql, language="sql")
        if scan_bytes:
            mb = scan_bytes / (1024 * 1024)
            st.caption(f"Data scanned: {mb:.2f} MB")


def render_error(message: str) -> None:
    st.error(f"**Error:** {message}")
