import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import openai
from dotenv import load_dotenv, find_dotenv

# Model, Prompts and Parser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.llms import OpenAI

# Chains
import pandas as pd
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# Question and Answer
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

# Evaluation
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
import langchain

from IPython.display import display, Markdown

# Agents
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent, tool, AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from datetime import date


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]
