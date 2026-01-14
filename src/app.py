import streamlit as st
import requests

st.set_page_config(page_title="FactGuard AI", layout="wide")

st.title("🛡️ FactGuard AI")
st.subheader("Agentic Healthcare Verification")

user_input = st.text_area("Enter a medical statement to verify:", height=100)

if st.button("Run Audit"):
    if user_input:
        with st.spinner("Researcher searching Qdrant... Auditor verifying..."):
            try:
                # Calls your FastAPI main.py
                res = requests.post("http://localhost:8001/analyze", params={"claim": user_input})
                data = res.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info("Researcher Findings")
                    st.write(data["analysis"])
                
                with col2:
                    st.success("Audit Results")
                    status = "CLEAN" if data["audit_passed"] else "❌ FLAG FOR REVISION"
                    st.metric("Final Status", status)
            except Exception as e:
                st.error("Is your FastAPI backend running? Run 'uv run python src/main.py' first.")