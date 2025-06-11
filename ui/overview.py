"""
Overview tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import config

def render_overview_tab(df, stats):
    """
    Render the overview tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    stats (dict): Dictionary with basic statistics
    """
    st.header("Chat Overview")
    
    if not stats:
        st.warning("No data to display.")
        return
    
    # Display basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", f"{stats['total_messages']:,}")
        st.metric("Total Days", f"{stats['date_range']:,}")
    
    with col2:
        st.metric("Avg. Messages/Day", f"{stats['avg_messages_per_day']:.1f}")
        st.metric("Peak Hour", f"{stats['peak_hour']}:00")
    
    with col3:
        st.metric("Most Active User", stats['most_active_user'])
        st.metric("Messages", f"{stats['most_active_user_count']:,}")
    
    # Message distribution by user
    st.subheader("Message Distribution by User")
    fig = px.pie(
        values=stats['messages_per_user'].values,
        names=stats['messages_per_user'].index,
        title="Messages per User",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Message activity by hour of day
    st.subheader("Activity by Hour of Day")
    fig = px.line(
        x=stats['hourly_activity'].index,
        y=stats['hourly_activity'].values,
        markers=True,
        title="Messages by Hour of Day",
        labels={'x': 'Hour', 'y': 'Number of Messages'}
    )
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Message activity by day of week
    st.subheader("Activity by Day of Week")
    fig = px.bar(
        x=stats['day_of_week_counts'].index,
        y=stats['day_of_week_counts'].values,
        title="Messages by Day of Week",
        labels={'x': 'Day', 'y': 'Number of Messages'},
        color_discrete_sequence=[config.CHART_COLORS['blue']]
    )
    st.plotly_chart(fig, use_container_width=True)
