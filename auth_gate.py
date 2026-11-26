"""Shared-password gate for the hosted (Azure) app.

Active only when the APP_PASSWORD environment variable is set, so local runs are unaffected.
"""
import hmac
import os

import streamlit as st


def require_password(title="AGEMIS: Agentic Horizon Scanning and Scenario Generation"):
    expected = os.getenv("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return
    st.title(title)
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()
