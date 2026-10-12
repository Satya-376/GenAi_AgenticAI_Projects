import openai

openai.api_key = "Enter you Open Ai api Key here"

def retriever_info(query):
    # Dummy implementation for example purposes
    return "about elon mask"

def rag_query(query):
    retrieved_info = retriever_info(query)  # Call to the retriever_info function
    augmented_prompt = f"User query: {query}. Retrieved information: {retrieved_info}"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5",  # Use the correct model
        messages=[
            {"role": "user", "content": augmented_prompt}
        ],
        max_tokens=150,
        temperature=0.2,
    
    )
    
    return response.choices[0].message['content'].strip()

# Example usage
query = "Tell me about the elon musk"
response = rag_query(query)
print(response)

