import streamlit as st
from openai import OpenAI
import os 

client=OpenAI(api_key="Enter your OpenAi Api key here")
#openai.api_key = ""

def retriever_info(query):
    # Dummy implementation for example purposes
    return "about prime minister of india"

def rag_query(query):
    retrieved_info = retriever_info(query)  # Call to the retriever_info function
    augmented_prompt = f"User query: {query}. Retrieved information: {retrieved_info}"
    
    response = client.chat.completions.create(
        model="gpt-6-luna",  # Use the correct model
        messages=[
            {"role": "user", "content": augmented_prompt}
        ],
        #max_tokens=150,
        #temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("RAG Prompt Query Interface")
user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        response = rag_query(user_input)
        st.write("Response:", response)
    else:
        st.write("Please enter a query.")

