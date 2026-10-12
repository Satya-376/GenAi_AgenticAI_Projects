from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Create prompt
prompt = PromptTemplate.from_template(
    "Answer the following question clearly and briefly:\n\n{question}"
)

# Create OpenAI chat model
model = ChatOpenAI(
    model="gpt-6-luna"
)

# Convert AIMessage to plain string
parser = StrOutputParser()

# Build the chain
chain = prompt | model | parser

# Execute the chain
result = chain.invoke({
    "question": "What is the capital of China?"
})

# Display the result
print(result)

