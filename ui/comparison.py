"""
Comparison tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from modules.time_series import generate_time_series
from modules.utils import get_freq_label

def render_comparison_tab(df, qualified_users, stats, time_freq):
    """
    Render the user comparison tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    qualified_users (list): List of users with minimum message count
    stats (dict): Dictionary with basic statistics
    time_freq (str): Time frequency for analysis ('D', 'W', or 'M')
    """
    st.header("User Comparison")
    
    # Allow selecting up to 5 users to compare
    max_users = min(config.MAX_USERS_TO_COMPARE, len(qualified_users))
    compare_users = st.multiselect(
        f"Select up to {max_users} users to compare",
        options=qualified_users,
        default=qualified_users[:min(2, len(qualified_users))]
    )
    
    if len(compare_users) > max_users:
        st.warning(f"Please select up to {max_users} users for comparison.")
        compare_users = compare_users[:max_users]
    
    if len(compare_users) >= 2:
        # Message count comparison
        st.subheader("Message Count Comparison")
        user_counts = stats['messages_per_user'][stats['messages_per_user'].index.isin(compare_users)]
        fig = px.bar(
            x=user_counts.index,
            y=user_counts.values,
            title="Message Count by User",
            labels={'x': 'User', 'y': 'Number of Messages'},
            color=user_counts.index,
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Message length comparison
        st.subheader("Average Message Length Comparison")
        user_avg_lengths = stats['avg_length_per_user'][stats['avg_length_per_user'].index.isin(compare_users)]
        fig = px.bar(
            x=user_avg_lengths.index,
            y=user_avg_lengths.values,
            title="Average Message Length by User",
            labels={'x': 'User', 'y': 'Average Characters per Message'},
            color=user_avg_lengths.index,
            text_auto='.1f'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Side-by-side hourly activity comparison
        st.subheader("Hourly Activity Pattern Comparison")
        fig = make_subplots(rows=1, cols=len(compare_users), 
                          subplot_titles=[f"{user}" for user in compare_users],
                          shared_yaxes=True)
        
        for i, user in enumerate(compare_users):
            user_data = df[df['author'] == user].copy()
            user_data.loc[:, 'hour'] = user_data['date'].dt.hour
            hourly_counts = user_data.groupby('hour').size().reindex(range(24), fill_value=0)
            
            fig.add_trace(
                go.Bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    name=user,
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            height=400,
            title_text="Hourly Activity Comparison",
            showlegend=False
        )
        
        for i in range(len(compare_users)):
            fig.update_xaxes(title_text="Hour", row=1, col=i+1)
            if i == 0:
                fig.update_yaxes(title_text="Number of Messages", row=1, col=i+1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series comparison
        st.subheader("Activity Over Time Comparison")
        ts_comparison = generate_time_series(df, compare_users, time_freq)
        
        # Get human readable frequency label
        freq_label = get_freq_label(time_freq)
        
        fig = px.line(
            ts_comparison,
            x=ts_comparison.index,
            y=ts_comparison.columns,
            title=f"{freq_label} Message Activity Comparison",
            labels={'x': 'Date', 'y': 'Number of Messages', 'variable': 'User'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least 2 users to compare.")
