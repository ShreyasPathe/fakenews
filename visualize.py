import streamlit as st
import plotly.graph_objects as go
import time

# Dummy prediction result
prediction_result = {
    "naive_bayes": 89.5,
    "svm": 76.3,
    "cnn": 92.1,
    "final": 85.0
}

def animate_bar_chart(result):
    y_labels = ["Naive Bayes", "SVM", "CNN", "Final"]
    keys = ["naive_bayes", "svm", "cnn", "final"]
    target_values = [result.get(key, 0.0) for key in keys]

    # Set default grey for all except final
    colors = ['#D3D3D3'] * 3
    colors.append('#00FF00' if result["final"] > 40 else '#FF0000')  # Green or Red

    progress_values = [0.0] * 4
    steps = 50

    chart_placeholder = st.empty()

    for step in range(1, steps + 1):
        current_values = [target * step / steps for target in target_values]

        fig = go.Figure(go.Bar(
            x=y_labels,
            y=current_values,
            text=[f"{val:.1f}%" for val in current_values],
            textposition='outside',
            marker_color=colors
        ))

        fig.update_layout(
            xaxis=dict(title='Model'),
            yaxis=dict(range=[0, 100], title='Real (%)'),
            height=400,
            margin=dict(l=100, r=50, t=30, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.02)

def animate_pie_chart(result):
    confidence = result["final"]
    steps = 50
    chart_placeholder = st.empty()

    for step in range(1, steps + 1):
        real_val = confidence * step / steps
        fake_val = 100 - real_val
        values = [real_val, fake_val]
        colors = ['#00FF00', '#FF0000'] if confidence > 40 else ['#FF0000', '#00FF00']

        fig = go.Figure(data=[go.Pie(
            labels=['Real', 'Fake'],
            values=values,
            hole=0.3,
            marker=dict(colors=colors)
        )])

        fig.update_layout(title="Final Verdict Distribution")
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.02)

# Streamlit UI
st.title("📰 Fake News Detection Visualization")

if st.button("Visualize", key="unique_visualize_button"):
    st.subheader("Choose Visualization Type:")
    chart_type = st.radio(
        "Select the type of chart to visualize prediction results:",
        ["Bar Chart", "Pie Chart"],
        key="chart_selector"
    )

    if chart_type == "Bar Chart":
        animate_bar_chart(prediction_result)
    elif chart_type == "Pie Chart":
        animate_pie_chart(prediction_result)

    st.info("🟩 If the 'Final' bar is green, the news is most likely real. "
            "🟥 If red, it is most likely fake.")
