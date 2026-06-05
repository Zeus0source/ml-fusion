import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
.test-bar {
    position: fixed;
    top: 0; left: 0;
    width: 100%;
    height: 52px;
    background: red;
    z-index: 99999;
    display: flex;
    align-items: center;
    padding: 0 20px;
    color: white;
    font-size: 20px;
}
</style>
<div class="test-bar">NAVBAR TEST</div>
""", unsafe_allow_html=True)

st.write("hello")