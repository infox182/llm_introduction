import warnings
import os
import getpass
import requests
import json

from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding

from typing import List, Optional

warnings.filterwarnings("ignore")


# # 1. GigaChat
# Define GigaChat throw langchain.chat_models

def get_giga(giga_key: str) -> GigaChat:
    return GigaChat(credentials=giga_key, verify_ssl_certs=False)

def test_giga():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)


# # 2. Prompting
# ### 2.1 Define classic prompt

# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List:
    system_message = SystemMessage(content="Вы - полезный ассистент, который отвечает на вопросы пользователя.")
    human_message = HumanMessage(content=user_content)
    
    return [system_message, human_message]


# Let's check how it works
def tes_prompt():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    user_content = 'Hello!'
    prompt = get_prompt(user_content)
    res = giga(prompt)
    print (res.content)

# ### 3. Define few-shot prompting

# Implement a function to build a few-shot prompt to count even digits in the given number.
# The answer should be in the format 'Answer: The number {number} consist of {text} even digits.', 
# for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    # Создаем примеры для few-shot обучения
    examples = [
        {"number": "1234", "answer": "Answer: The number 1234 consist of two even digits."},
        {"number": "5678", "answer": "Answer: The number 5678 consist of two even digits."},
        {"number": "11223344", "answer": "Answer: The number 11223344 consist of four even digits."},
        {"number": "9753", "answer": "Answer: The number 9753 consist of zero even digits."}
    ]
    
    prompt_text = "Посчитайте количество четных цифр в числе и ответьте в указанном формате.\n\n"
    
    for example in examples:
        prompt_text += f"Число: {example['number']}\n{example['answer']}\n\n"
    
    prompt_text += f"Число: {number}\n"

    return [HumanMessage(content=prompt_text)]

# Let's check how it works
def test_few_shot():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    number = '62388712774'
    prompt = get_prompt_few_shot(number)
    res = giga.invoke(prompt)
    print (res.content)

# # 4. Llama_index
# Implement your own class to use llama_index.
# You need to implement some code to build llama_index across your own documents.
# For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt="""
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        documents = SimpleDirectoryReader(path_to_data).load_data()
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        )
        
        # Создаем сервисный контекст с указанной LLM и embeddings
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model
        )
        
        # Создаем индекс на основе загруженных документов
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        
        # Создаем query engine с указанным системным промптом
        self.query_engine = self.index.as_query_engine(
            system_prompt=self.system_prompt
        )

    def query(self, user_prompt: str) -> str:
        response = self.query_engine.query(user_prompt)
        return str(response)


# Let's check
def test_llama_index():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query('what is attention is all you need?')
    print (res)


if __name__ == "__main__":
    test_few_shot()
    tes_prompt()
    test_llama_index()
