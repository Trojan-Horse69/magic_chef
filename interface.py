import streamlit as st
from chain import agent_executor
from dotenv import load_dotenv
import os

load_dotenv()
fireworks_api_key = os.environ['FIREWORKS_API_KEY']
nomic_api_key = os.environ['NOMIC_API_KEY']


st.title('Chef Lazio')
user_prompt = st.text_input('Enter a comma-seperated list of ingredients')

if st.button("Generate") and user_prompt:
    with st.spinner("Generating...."):
        output = agent_executor.invoke({"input": user_prompt})
        st.write(output)
