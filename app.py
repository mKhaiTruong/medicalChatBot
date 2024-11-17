from flask import Flask, render_template, jsonify, request
from src.helper import load_embeddings
from langchain_ollama import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

MAX_TOKENS = 2048
llm = OllamaLLM(model="llama3.2", max_tokens=MAX_TOKENS, temperature=0.5)
embedding = load_embeddings()


doc_search = PineconeVectorStore.from_existing_index(
    index_name="med",
    embedding=embedding
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever = doc_search.as_retriever(search_type='similarity', search_kwargs={"k": 3})
ques_ans_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, ques_ans_chain)


@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/get', methods=['POST'])
def chat():
    try:
        msg = request.form['msg']
        print(f'Input = {msg}')

        res = rag_chain.invoke({"input": msg})
        print(f"Response: {res['answer']}")
        return str(res['answer'])
    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred", 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
