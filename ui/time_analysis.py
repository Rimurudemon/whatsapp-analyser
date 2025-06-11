"""
Time Analysis tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import config
from modules.time_series import generate_time_series
from modules.utils import get_freq_label, get_window_size

def render_time_analysis_tab(df, qualified_users, time_freq):
    """
    Render the time analysis tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    qualified_users (list): List of users with minimum message count
    time_freq (str): Time frequency for analysis ('D', 'W', or 'M')
    """
    st.header("Time Series Analysis")
    
    # User selector for time series (multiselect)
    selected_users_ts = st.multiselect(
        "Select users to include (leave empty for all)",
        options=qualified_users,
        default=[qualified_users[0]] if qualified_users else []
    )
    
    # Generate time series
    with st.spinner("Generating time series..."):
        if selected_users_ts:
            time_series_df = generate_time_series(df, selected_users_ts, time_freq)
        else:
            time_series_df = generate_time_series(df, freq=time_freq)
    
    if time_series_df.empty:
        st.warning("Not enough data for time series analysis.")
        return
    
    # Plotting the time series
    st.subheader("Message Activity Over Time")
    
    # Convert frequency to human-readable label
    freq_label = get_freq_label(time_freq)
    
    fig = px.line(
        time_series_df,
        x=time_series_df.index,
        y=time_series_df.columns,
        title=f"{freq_label} Message Activity",
        labels={'x': 'Date', 'y': 'Number of Messages', 'variable': 'User'}
    )
    fig.update_layout(legend_title_text='User')
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling average
    st.subheader("Activity Trend (Rolling Average)")
    window_size = get_window_size(time_freq)
    
    if len(time_series_df) > window_size:
        rolling_df = time_series_df.rolling(window=window_size).mean()
        
        fig = px.line(
            rolling_df,
            x=rolling_df.index,
            y=rolling_df.columns,
            title=f"{window_size}-{freq_label.lower()} Rolling Average",
            labels={'x': 'Date', 'y': 'Number of Messages', 'variable': 'User'}
        )
        fig.update_layout(legend_title_text='User')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Not enough data points for a {window_size}-{freq_label.lower()} rolling average.")
