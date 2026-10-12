from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.chat_models import ChatOpenAI

import os
os.environ["OPENAI_API_KEY"] = "Enter your OpenAi key here"
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","I am chatbot. I am hear to assist you. Please type your queries"),
        ("user","Question:{question}")
    ]
)

# openAI LLm
llm=ChatOpenAI(model="gpt-6-luna", temperature=1)
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

response = chain.invoke('ask me anything: what is difference between genrerative ai vs agetic ai')

print(response)