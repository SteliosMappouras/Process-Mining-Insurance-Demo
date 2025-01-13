import streamlit as st
import pm4py
import pandas as pd
import numpy as np
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.statistics.variants.log import get as variants_get  # Fixed import for trace statistics
import tempfile

# Streamlit UI Setup
st.title("🚀 Process Mining in Insurance - Enhanced Claims Processing Demo")
st.sidebar.header("Upload Event Log File")

# Upload event log CSV
uploaded_file = st.sidebar.file_uploader("Upload an event log CSV file", type="csv")

if uploaded_file:
    # Load the event log with user roles and demographics
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.write("This table shows the raw event log data. Each row represents an activity in a claim's lifecycle.")
    st.write(df.head())

    # Verify columns existence
    required_columns = {'case_id', 'activity', 'timestamp', 'user_role', 'customer_age'}
    if not required_columns.issubset(df.columns):
        st.error("The file must contain 'case_id', 'activity', 'timestamp', 'user_role', and 'customer_age' columns!")
    else:
        # Convert to event log
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        event_log = pm4py.format_dataframe(df, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
        event_log = log_converter.apply(event_log)

        # ------- Alpha Miner -------
        st.subheader("🔎 Alpha Miner Process Discovery")
        st.write("Alpha Miner discovers the basic process structure from the event log using strict patterns.")
        net, initial_marking, final_marking = alpha_miner.apply(event_log)
        gviz_alpha = pn_visualizer.apply(net, initial_marking, final_marking)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pn_visualizer.save(gviz_alpha, temp_file.name)
            st.image(temp_file.name)

        # ------- Heuristics Miner -------
        st.subheader("📊 Heuristics Miner Process Discovery")
        st.write("Heuristics Miner provides a more flexible process discovery approach, allowing for noise and exceptions.")
        net, im, fm = heuristics_miner.apply(event_log)
        gviz_heuristics = pn_visualizer.apply(net, im, fm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pn_visualizer.save(gviz_heuristics, temp_file.name)
            st.image(temp_file.name)

        # Inductive Miner Discovery
        st.subheader("🌲 Inductive Miner Process Discovery")
        st.write("Inductive Miner creates a hierarchical model that represents the process in a structured way.")
        process_tree = inductive_miner.apply(event_log)
        net, im, fm = pt_converter.apply(process_tree)
        gviz_inductive = pn_visualizer.apply(net, im, fm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pn_visualizer.save(gviz_inductive, temp_file.name)
            st.image(temp_file.name)

        # ------- Bottleneck Detection -------
        st.subheader("⏳ Bottleneck Analysis")
        st.write("This section highlights the cases with the longest durations, which are potential bottlenecks.")
        df['case_duration'] = df.groupby('case_id')['timestamp'].transform(lambda x: x.max() - x.min())
        bottleneck_cases = df.sort_values(by='case_duration', ascending=False).head(5)
        st.write("Cases with the Longest Durations (Potential Bottlenecks)")
        st.write(bottleneck_cases[['case_id', 'case_duration']])

        # ------- Demographics Insights -------
        st.subheader("📈 Demographics Insights")
        st.write("Analyze customer demographics such as age distribution to understand patterns and trends.")
        avg_age = df['customer_age'].mean()
        age_distribution = df['customer_age'].value_counts().sort_index()
        st.write(f"**Average Customer Age:** {avg_age:.2f} years")
        st.bar_chart(age_distribution)

        # ------- User Role Insights -------
        st.subheader("👥 User Role Analysis")
        st.write("This chart shows the number of activities handled by each user role in the claims process.")
        user_role_counts = df['user_role'].value_counts()
        st.write("**Number of Activities Handled by Each Role**")
        st.bar_chart(user_role_counts)

        # ------- Trace Variants Analysis -------
        st.subheader("🛤️ Trace Variants Analysis")
        st.write("Trace variants represent different paths taken by cases in the process.")
        variants_count = variants_get.get_variants(event_log)
        sorted_variants = sorted(variants_count.items(), key=lambda x: len(x[1]), reverse=True)
        st.write("Top Process Variants:")
        for variant, cases in sorted_variants[:5]:
            st.write(f"**Variant:** {variant} | **Count:** {len(cases)}")

        # ------- Replay Fitness -------
        st.subheader("🎯 Replay Fitness")
        st.write("Replay Fitness evaluates how well the discovered process model aligns with the actual event log.")
        fitness = replay_fitness.apply(event_log, net, initial_marking, final_marking)
        st.write(f"**Replay Fitness:** {fitness['log_fitness']}")

        # ------- Key Takeaways Section -------
        st.subheader("📌 Key Insights and Takeaways")
        st.markdown("""
        - **Bottlenecks Identified:** Longest-running cases highlighted above.
        - **Role Insights:** Identify which roles perform the most activities.
        - **Demographic Patterns:** Visualize age distribution.
        - **Process Accuracy:** Replay fitness provides model quality insights.
        """)
else:
    st.warning("📥 Please upload an event log file to get started!")

st.sidebar.info("Upload an event log file with columns: `case_id`, `activity`, `timestamp`, `user_role`, `customer_age`.")
