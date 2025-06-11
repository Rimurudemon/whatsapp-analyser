"""
Chat Patterns tab UI component for WhatsApp Chat Analyzer
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from modules.time_series import analyze_chat_patterns

def render_chat_patterns_tab(df, qualified_users):
    """
    Render the chat patterns tab UI
    
    Parameters:
    df (DataFrame): DataFrame with parsed messages
    qualified_users (list): List of users with minimum message count
    """
    st.header("Chat Patterns Analysis")

    # Response time analysis
    st.subheader("Response Time Analysis")

    # Select users for response time analysis
    response_users = st.multiselect(
        "Select users to analyze response times (max 3)",
        options=qualified_users,
        default=qualified_users[:min(2, len(qualified_users))]
    )

    if len(response_users) >= 2 and len(response_users) <= 3:
        # Calculate conversation gap for new conversations
        conversation_gap_hours = st.slider("Hours of inactivity to define a new conversation", 1, 12, 3)
        chat_patterns = analyze_chat_patterns(df, conversation_gap_hours)
        
        # Extract response times for selected users only
        response_analysis = {}
        for key, times in chat_patterns['response_times'].items():
            # Check if the key contains both selected users
            users_in_key = [user for user in response_users if user in key]
            if len(users_in_key) == 2:  # Both users are in the key
                response_analysis[key] = times

        if response_analysis:
            # Calculate average response times
            avg_response_times = {k: np.mean(v) for k, v in response_analysis.items()}

            # Create a DataFrame for plotting
            response_df = pd.DataFrame({
                'Direction': list(avg_response_times.keys()),
                'Avg Response Time (min)': list(avg_response_times.values())
            })

            fig = px.bar(
                response_df,
                x='Direction',
                y='Avg Response Time (min)',
                title="Average Response Time (minutes)",
                color='Direction',
                text_auto='.1f'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Distribution of response times
            st.subheader("Response Time Distribution")

            # Create figure with 1 row and up to 3 columns
            num_pairs = len(response_analysis)
            fig = make_subplots(rows=1, cols=num_pairs, 
                               subplot_titles=list(response_analysis.keys()))

            for i, (direction, times) in enumerate(response_analysis.items()):
                # Filter out extreme values (>95th percentile) for better visualization
                upper_limit = np.percentile(times, 95)
                filtered_times = [t for t in times if t <= upper_limit]
                
                fig.add_trace(
                    go.Histogram(
                        x=filtered_times,
                        name=direction,
                        marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    ),
                    row=1, col=i+1
                )

            fig.update_layout(
                height=400,
                title_text="Response Time Distribution (minutes)",
                showlegend=False
            )

            for i in range(num_pairs):
                fig.update_xaxes(title_text="Minutes", row=1, col=i+1)
                if i == 0:
                    fig.update_yaxes(title_text="Frequency", row=1, col=i+1)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough conversation data between these users to analyze response times.")
    
    elif len(response_users) > 3:
        st.warning("Please select at most 3 users for response time analysis.")
    else:
        st.info("Please select at least 2 users to analyze response patterns.")

    # Conversation Initiator Analysis
    st.subheader("Conversation Initiator Analysis")

    # Get conversation initiators if not already calculated
    if 'chat_patterns' not in locals():
        conversation_gap_hours = st.slider("Hours of inactivity to define a new conversation", 1, 12, 3)
        chat_patterns = analyze_chat_patterns(df, conversation_gap_hours)
        
    initiators = chat_patterns['initiators']
    total_conversations = len(initiators)

    if total_conversations > 0:
        # Convert to DataFrame for plotting
        initiator_df = pd.DataFrame({
            'User': initiators.index,
            'Count': initiators.values,
            'Percentage': (initiators.values / initiators.sum() * 100).round(1)
        })

        # Create the plot
        fig = px.pie(
            initiator_df,
            values='Count',
            names='User',
            title=f"Conversation Initiators (Total: {total_conversations} conversations)",
            hover_data=['Percentage'],
            labels={'Percentage': '% of conversations'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to analyze conversation initiators.")

    # Activity Heatmap
    st.subheader("Activity Heatmap")

    # Get activity heatmap data
    activity_heatmap = chat_patterns['activity_heatmap']
    
    # Replace day numbers with names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity_heatmap.index = day_names

    # Create the heatmap
    fig = px.imshow(
        activity_heatmap,
        labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
        x=[f"{i:02d}:00" for i in range(24)],
        y=day_names,
        color_continuous_scale="Viridis",
        title="Activity Heatmap (Day of Week vs Hour of Day)"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
