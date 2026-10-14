import streamlit as st
import pandas as pd
import numpy as np
from summarizer import (
    summarize, 
    remove_stopwords_and_clean, 
    compute_two_document_similarity
)

st.set_page_config(
    page_title="Document Summarizer | IR Mini Project",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .summary-box {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1F2937;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_TEXT = """Artificial intelligence is transforming healthcare by helping doctors diagnose diseases, analyze medical images, and improve patient care. AI systems can process large amounts of medical data and identify patterns that may be difficult for humans to detect. As a result, hospitals and research institutions are increasingly adopting artificial intelligence technologies.

One of the most important applications of artificial intelligence in healthcare is disease diagnosis. Machine learning algorithms can analyze X-rays, CT scans, and MRI images to identify abnormalities and assist doctors in detecting diseases at an early stage. Early diagnosis can help medical professionals make better treatment decisions and improve patient outcomes.

Artificial intelligence is also improving patient monitoring and personalized treatment. AI-powered systems can analyze patient records, monitor vital signs, and identify potential health risks. By examining a patient's medical history and current condition, these systems can help doctors develop more personalized treatment plans.

Another important application of AI is the automation of routine hospital tasks. Intelligent systems can assist with appointment scheduling, medical record management, and administrative documentation. Automation reduces repetitive work and allows healthcare professionals to spend more time caring for patients.

Despite these benefits, the use of artificial intelligence in healthcare presents several challenges. Medical AI systems require high-quality data to produce reliable results. Incomplete or biased datasets can lead to inaccurate predictions and unequal treatment recommendations. Protecting patient privacy and securing confidential medical records are also essential.

Artificial intelligence should support rather than completely replace healthcare professionals. Doctors must interpret AI-generated recommendations using their clinical knowledge and experience. Human supervision is particularly important when AI systems make predictions that could affect patient safety.

The successful adoption of artificial intelligence in healthcare depends on reliable technology, high-quality medical data, strong privacy protections, and collaboration between researchers and healthcare professionals. When implemented responsibly, AI can improve disease diagnosis, support personalized treatment, reduce administrative workloads, and contribute to better patient care."""

with st.sidebar:
    st.header("⚙️ Summarizer Settings")
    
    num_sentences = st.slider(
        "Summary Length (Sentences)", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="Number of top PageRank sentences to include in summary"
    )
    
    damping_factor = st.slider(
        "PageRank Damping Factor (d)", 
        min_value=0.50, 
        max_value=0.95, 
        value=0.85, 
        step=0.05,
        help="Probability of random sentence transition (standard = 0.85)"
    )

st.markdown('<div class="main-title">📝 IR Document Summarizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Extractive text summarization powered by <b>TF-IDF</b>, <b>Cosine Similarity</b>, and <b>PageRank (TextRank)</b> algorithm.</div>', 
    unsafe_allow_html=True
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    if st.button("✨ Load Sample Article"):
        st.session_state["input_text"] = SAMPLE_TEXT
with col_btn2:
    if st.button("🧹 Clear Text"):
        st.session_state["input_text"] = ""

input_text = st.text_area(
    "Enter or paste the document text below:",
    value=st.session_state.get("input_text", SAMPLE_TEXT),
    height=240,
    placeholder="Paste your article or document text here..."
)

col_run, _ = st.columns([2, 5])
with col_run:
    summarize_clicked = st.button("Generate Summary", type="primary")

if input_text.strip():
    result = summarize(input_text, num_sentences=num_sentences, damping_factor=damping_factor)

    st.markdown("---")
    st.subheader("📋 Generated Summary")

    if result["summary"]:
        st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Original Words", result["original_word_count"])
        with m2:
            st.metric("Summary Words", result["summary_word_count"])
        with m3:
            st.metric("Sentences Chosen", f"{len(result['top_indices'])} / {len(result['sentences'])}")
        with m4:
            st.metric("Compression Ratio", f"{result['compression_ratio']}%")

        st.markdown("### Information Retrieval Step-by-Step Breakdown")
        tab_rank, tab_prep = st.tabs([
            "PageRank Scores", 
            "Preprocessing & Stop Words", 
        ])

        with tab_rank:
            st.markdown("**Sentence Ranking via PageRank (Top 5)**")
            st.caption("Sentences with higher similarity to other key sentences receive higher PageRank scores.")
            
            table_data = []
            for item in result["ranked_sentences"][:5]:
                is_selected = "✅ Yes (In Summary)" if item["index"] in result["top_indices"] else "No"
                table_data.append({
                    "Rank": item["rank"],
                    "Sentence Index": f"Sentence {item['index'] + 1}",
                    "PageRank Score": f"{item['score']:.5f}",
                    "Included in Summary": is_selected,
                    "Sentence Text": item["text"]
                })
            df_ranks = pd.DataFrame(table_data)
            st.dataframe(df_ranks, hide_index=True)

        with tab_prep:
            st.markdown("**Pre-processing: Lowercasing & Stop Word Removal (Top 5 Sentences)**")
            st.caption("Common stop words are filtered out prior to TF-IDF vectorization.")
            
            prep_rows = []
            for i, s in enumerate(result["sentences"][:5]):
                cleaned = remove_stopwords_and_clean(s)
                prep_rows.append({
                    "Sentence #": f"S{i+1}",
                    "Original Sentence": s,
                    "After Stop Word Removal": cleaned
                })
            st.dataframe(pd.DataFrame(prep_rows), hide_index=True)

            
    else:
        st.warning("Please enter valid text to summarize.")
else:
    st.info("Enter text above or click **Load Sample Article** and press **Generate Summary**.")
