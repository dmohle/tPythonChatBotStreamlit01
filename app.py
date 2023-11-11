import os

from langchain.memory import ConversationBufferMemory

from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"] = apikey

# pip install:
#   streamlit
#   langchain
#   openai
#   wikipedia
#   chromadb
#   tiktoken

st.title(" 🦜️🔗 Welcome to my AI Chatbot! 🦜️🔗 ")
prompt_from_user = st.text_input("You know the question...")

# Prompt templates.
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write me a youtube video title about {topic}"
)

# Prompt templates.
script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="""Write me a youtube script based on this title about {title}
        and also collect this wikipedia research: {wikipedia_research}
             """
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# llms
my_llm = OpenAI(temperature=0.5)
title_chain = LLMChain(llm=my_llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory)

script_chain = LLMChain(llm=my_llm, prompt=script_template, verbose=True,
                        output_key="script", memory=script_memory)

wiki = WikipediaAPIWrapper()

# Output answer to streamlit server screen
# did the user type something into the text area?
if prompt_from_user:
    # Use the llm() function to send the prompt to ChatGPT
    # what model ??
    title = title_chain.run(prompt_from_user)
    wiki_research = wiki.run(prompt_from_user)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    #   st.write(answer["title"])
    st.write(script)

    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("SCript History"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Research"):
        st.info(wiki_research)