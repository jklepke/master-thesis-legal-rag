import json 
import sys 
import os
import streamlit as st 

# Argument von der Kommandozeile lesen
if len(sys.argv) > 1:
    json_path = sys.argv[1]
else:
    json_path = "test_evalset_extended.json"

def run_manual_evaluation(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    st.set_page_config(layout="wide")

    st.title("Manual Evaluation Interface")


    if "index" not in st.session_state:
        st.session_state.index = 0
    if "results" not in st.session_state:
        st.session_state.results = []

    if st.session_state.index >= len(dataset): 
        st.success("✅ All entries evaluated!")

        # Lokale Speicherung in Ordner deiner Wahl
        output_dir = "eval_results"  # z. B. Unterordner im Projekt
        os.makedirs(output_dir, exist_ok=True)  # Ordner erstellen, falls nicht vorhanden

        output_path = os.path.join(output_dir, "manual_eval_results.json")
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(st.session_state.results, f_out, ensure_ascii=False, indent=2)

        # Optional in Streamlit anzeigen
        st.info(f"📁 Results saved to `{output_path}`")

        # Download-Button anbieten
        st.download_button("📥 Download Results", 
            json.dumps(st.session_state.results, indent=2, ensure_ascii=False), 
            file_name="manual_eval_results.json")
        return

    entry = dataset[st.session_state.index]

    st.markdown(f"**Query ID:** {entry['query_id']}")
    st.markdown(f"**Question Context:** {entry['question_context']}")
    st.markdown(f"**Question:** <div style='white-space: pre-wrap; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>{entry['query']}</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Generated Response**")
        st.markdown(f"<div style='white-space: pre-wrap; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>{entry['generated_response']}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Ground Truth**")
        st.markdown(f"<div style='white-space: pre-wrap; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>{entry['ground_truth']}</div>", unsafe_allow_html=True)


    st.subheader("Assign Scores (1–5)")
    with st.expander("View Full Evaluation Scale (1–5 Scale)"):
       st.markdown("""
| **Score** | **Answer Relevance** | **Faithfulness** | **Answer Correctness** |
|-----------|---------------|------------------|-----------------|
| **1**     | The answer is irrelevant and off-topic to the question or entirely misses the user’s intent. | The answer contains factual errors, hallucinations, or unsupported claims. | No meaningful overlap with the reference answer; key points are missing or contradicted. |
| **2**     | The answer is only marginally related to the question; it shows limited topical relevance. | Some parts are accurate, but major factual inaccuracies or fabricated content are present. | Limited alignment; only a few aspects of the reference are addressed, and several key elements are omitted. |
| **3**     | The answer is partially relevant but lacks completeness or includes some off-topic elements. | Mostly factually correct, but contains minor errors, ambiguities, or uncertain claims. | Partial overlap with the reference; key points are present but not fully developed or precise. |
| **4**     | The answer is largely relevant and addresses most of the question, with minor omissions. | Factually sound with only negligible issues or unclear phrasing. | High degree of alignment with the reference, though not a perfect match in all details. |
| **5**     | The answer is highly relevant, directly addresses the question, and provides helpful insights. | Fully factually accurate, free of hallucinations or unsupported statements. | Strong match with the reference in meaning, structure, and coverage of key points. |
""")

        

    col1, col2, col3 = st.columns(3)

    with col1:
        relevance = st.radio("**Answer Relevance**", [1, 2, 3, 4, 5], horizontal=True)

    with col2:
        accuracy = st.radio("**Faithfulness**", [1, 2, 3, 4, 5], horizontal=True)

    with col3:
        alignment = st.radio("**Answer Correctness**", [1, 2, 3, 4, 5], horizontal=True)

    comment = st.text_area("**Optional Comment**")


    col_back, col_submit = st.columns([1, 1])
    
    with col_back:
        if st.session_state.index > 0:
            if st.button("Go Back"):
                st.session_state.index -= 1
    
    with col_submit:
        if st.button("✅ Submit Evaluation"):
            st.session_state.results.append({
                "query_id": entry["query_id"],
                "question_context": entry["question_context"],
                "query": entry["query"],
                "generated_response": entry["generated_response"],
                "ground_truth": entry["ground_truth"],
                "Relevance": relevance,
                "Factual Accuracy": accuracy,
                "Answer Alignment": alignment,
                "Comment": comment
            })
            st.session_state.index += 1

# App starten
run_manual_evaluation(json_path)