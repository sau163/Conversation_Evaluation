"""
Updated Streamlit UI for ML-Based Conversation Evaluation System
Uses the new production pipeline with retrieval + ML evaluator
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.production_pipeline import ProductionEvaluator

# Page config
st.set_page_config(
    page_title="Conversation Evaluation Benchmark",
    page_icon="💬",
    layout="wide"
)

# Initialize
if 'evaluator' not in st.session_state:
    with st.spinner("🚀 Loading ML-based evaluator..."):
        st.session_state.evaluator = ProductionEvaluator()

evaluator = st.session_state.evaluator

# Header
st.title("💬 Conversation Evaluation Benchmark")
st.markdown("**ML-Based System | No Prompting | Scales to 5000+ Facets**")

# Constraints badges
col1, col2, col3 = st.columns(3)
with col1:
    st.success("✅ No One-Shot Prompting")
with col2:
    st.success("✅ Open-Weights ≤16B")
with col3:
    st.success("✅ Scales to 5000+ Facets")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎯 System Info")
    stats = evaluator.retriever.get_statistics()
    st.metric("Total Facets", stats['total_facets'])
    st.metric("Categories", stats['categories'])

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Evaluate", "📊 Results", "🔍 Facets"])

# Tab 1: Evaluate
with tab1:
    st.header("Evaluate New Conversation")
    
    num_turns = st.number_input("Number of turns", 2, 6, 4)
    
    turns = []
    for i in range(num_turns):
        col1, col2 = st.columns([1, 4])
        with col1:
            speaker = st.selectbox(f"Turn {i+1}", ["User", "AI"], key=f"sp_{i}")
        with col2:
            text = st.text_area(f"Text", key=f"tx_{i}", height=60)
        
        if text:
            turns.append({"turn": i+1, "speaker": speaker, "text": text})
    
    if st.button("🚀 Evaluate", type="primary"):
        if len(turns) >= 2:
            with st.spinner("Evaluating..."):
                conv = {
                    "conversation_id": "demo",
                    "scenario": "Demo",
                    "turns": turns
                }
                result = evaluator.evaluate_conversation(conv)
                
                st.success("✅ Complete!")
                
                for turn in result['turns']:
                    with st.expander(f"Turn {turn['turn_number']}"):
                        st.write(f"**{turn['speaker']}:** {turn['text']}")
                        
                        scores_df = pd.DataFrame([
                            {'Facet': f, 'Score': d['score'], 'Category': d['category']}
                            for f, d in turn['facet_scores'].items()
                        ]).sort_values('Score', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Score", f"{scores_df['Score'].mean():.2f}")
                        with col2:
                            st.metric("Facets Evaluated", len(scores_df))
                        
                        st.dataframe(scores_df.head(10))

# Tab 2: Results
with tab2:
    st.header("Sample Results")
    
    results_file = Path("data/results/evaluation_results.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        st.success(f"Loaded {len(results)} conversations")
        
        selected_idx = st.selectbox(
            "Select conversation",
            range(len(results)),
            format_func=lambda x: f"Conv {results[x]['conversation_id']}: {results[x]['scenario']}"
        )
        
        result = results[selected_idx]
        
        st.metric("Total Turns", len(result['turns']))
        
        for turn in result['turns']:
            with st.expander(f"Turn {turn['turn_number']}: {turn['speaker']}"):
                st.write(turn['text'])
                
                scores = pd.DataFrame([
                    {'Facet': f, 'Score': d['score']}
                    for f, d in turn['facet_scores'].items()
                ])
                
                st.dataframe(scores.sort_values('Score', ascending=False).head(10))
    else:
        st.warning("No results found. Run evaluation first.")

# Tab 3: Facets
with tab3:
    st.header("Explore Facets")
    
    facets_df = evaluator.retriever.facets_df
    
    categories = ['All'] + sorted(facets_df['category'].unique().tolist())
    selected = st.selectbox("Category", categories)
    
    if selected != 'All':
        facets_df = facets_df[facets_df['category'] == selected]
    
    st.metric("Facets", len(facets_df))
    st.dataframe(facets_df[['facet_name', 'category', 'description']], height=400)

st.markdown("---")
st.markdown("**v2.0 | ML-Based | No Prompting | Production Ready**")
