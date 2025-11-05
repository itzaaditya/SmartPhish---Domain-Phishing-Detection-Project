# frontend_app.py
import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI-Based Phishing Domain Detector</h1>
    <p>Intelligent System to Detect Phishing Domains</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Enter a URL below to check its legitimacy using our trained machine learning model.  
The system analyzes 60+ features to determine if a website is safe or potentially malicious.
""")

# API Configuration
API_URL = "http://127.0.0.1:5000/predict"

# Main input area
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    url_input = st.text_input(
        "🔗 Enter URL to Analyze",
        placeholder="e.g., https://www.example.com",
        help="Enter the complete URL including http:// or https://"
    )
    
    analyze = st.button("🔍 Analyze Domain", use_container_width=True)
    
    # Example URLs
    with st.expander("💡 Try Example URLs"):
        st.markdown("""
        **Legitimate Sites:**
        - https://www.google.com
        - https://www.amazon.com
        - https://github.com
        
        **Suspicious Patterns (Test):**
        - http://192.168.1.1/login
        - https://secure-bank-login123.xyz
        """)

st.markdown("---")

# Result display area
if analyze and url_input:
    with st.spinner("🔎 Analyzing the URL... This may take a few seconds..."):
        try:
            # Check if URL has scheme
            if not url_input.startswith(('http://', 'https://')):
                st.warning("⚠️ URL should start with http:// or https://. Adding https:// automatically...")
                url_input = 'https://' + url_input
            
            # Make API request
            start_time = time.time()
            resp = requests.post(API_URL, json={"url": url_input}, timeout=30)
            elapsed_time = time.time() - start_time
            
            if resp.status_code != 200:
                st.error(f"❌ Server error: {resp.status_code} — {resp.text}")
            else:
                data = resp.json()
                
                # Extract prediction data
                prediction = data.get("prediction", None)
                confidence = data.get("confidence", None)
                probabilities = data.get("probabilities", [])
                
                if prediction is None:
                    st.error("❌ No prediction returned by server.")
                else:
                    # Display results based on prediction
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    
                    with col_b:
                        if str(prediction).lower() in ["phishing", "1"]:
                            st.error(f"### ⚠️ PHISHING DETECTED!")
                            st.markdown(f"""
                            <div style='background-color: #ff4b4b; padding: 1rem; border-radius: 8px; color: white;'>
                                <h4>⛔ This website is potentially dangerous!</h4>
                                <p><b>Confidence:</b> {confidence}%</p>
                                <p><b>Analysis Time:</b> {elapsed_time:.2f}s</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.success(f"### ✅ LEGITIMATE WEBSITE")
                            st.markdown(f"""
                            <div style='background-color: #00cc66; padding: 1rem; border-radius: 8px; color: white;'>
                                <h4>✓ This website appears to be safe</h4>
                                <p><b>Confidence:</b> {confidence}%</p>
                                <p><b>Analysis Time:</b> {elapsed_time:.2f}s</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show confidence meter
                        st.markdown("#### Prediction Confidence")
                        st.progress(confidence / 100)
                        
                        # Show probability breakdown
                        if probabilities and len(probabilities) >= 2:
                            st.markdown("#### Probability Breakdown")
                            classes = data.get("model_classes", ["Legitimate", "Phishing"])
                            
                            col_prob1, col_prob2 = st.columns(2)
                            with col_prob1:
                                st.metric(f"{classes[0]}", f"{probabilities[0]*100:.2f}%")
                            with col_prob2:
                                st.metric(f"{classes[1]}", f"{probabilities[1]*100:.2f}%")
                        
                        # Technical details
                        with st.expander("🔧 Show Technical Details"):
                            st.json(data)

        except requests.exceptions.ConnectionError:
            st.error("""
            ❌ **Connection Error!**
            
            Cannot connect to the backend server. Please ensure:
            1. Flask server is running (`python server.py`)
            2. Server is accessible at http://127.0.0.1:5000
            3. No firewall is blocking the connection
            """)
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The analysis is taking too long. Please try again.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

elif analyze and not url_input:
    st.warning("⚠️ Please enter a URL to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🔒 Phishing Detection System | Powered by Machine Learning</p>
    <p style='font-size: 0.8rem;'>Analyzes 60+ URL features including domain characteristics, HTML content, and security indicators</p>
</div>
""", unsafe_allow_html=True)