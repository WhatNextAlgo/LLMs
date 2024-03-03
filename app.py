import streamlit as st
import os

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] =os.getenv("HUGGINGFACEHUB_API_TOKEN")


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

#Function to return the response
def load_answer(question):
    llm = HuggingFaceEndpoint(repo_id = "google/gemma-7b", temperature=0.5,max_new_tokens =200)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer=llm_chain.invoke(question)
    return answer


#App UI starts here
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("Hey, I'm your InsightfulBot GPT")


#Gets the user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input=get_text()
response = load_answer(user_input)

submit = st.button('Generate')  

#If generate button is clicked
if submit:

    st.subheader("Answer:")

    st.write(response)
