import streamlit as st
from chain import recipe_agent_executor
from grocery_ai import grocery_agent_executor

st.title('Chef Lazio')

option = st.radio("Select Functionality:", ("Recipe Generator", "Grocery Assistant"))

if option == "Recipe Generator":
    user_prompt = st.text_input('Enter a comma-seperated list of ingredients')

    if st.button("Generate Recipe") and user_prompt:
        with st.spinner("Generating recipe...."):
            output = recipe_agent_executor.invoke({"input": user_prompt})
            st.write(output)
else:
    user_prompt = st.text_input('Enter the name of the recipe you need groceries for')

    if st.button("Generate Groceries") and user_prompt:
        with st.spinner("Generating groceries...."):
            output = grocery_agent_executor.invoke({"input": user_prompt})
            st.write(output)
