from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
openaijykey='./seoul_prompthon/.env'
load_dotenv(openaijykey)
openai_api_key = os.getenv("OPENAI_API_KEY_SSIMU")
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large", chunk_size=2000)
vectordb = Chroma(persist_directory="./seoul_prompthon/embedding_db/20240617_openai", embedding_function=embeddings)

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.1
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50}
    ), llm=llm
)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = ("""1. You are an expert who is familiar with Seoul’s policies for small business owners.
2. You must be honest about information you do not know. If you don't know, please say 'I can't find that information.'
3. Your answer must have a basis.
4. Do not repeat the instructions I wrote while adding numbers in your answer.
5. Please tell me the name of the document or website address where I can find more information. But don't lie.

Read the following explanation and write your answer using the context: {context}""")
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

result1 = conversational_rag_chain.invoke({"input":"나는 60대에 조그마한 편의점을 하고 있는데. 내가 무슨 지원을 받을 수 있을까?"}, config={"configurable": {"session_id": "test_v1"}})
result2 = conversational_rag_chain.invoke({"input":"나는 나이가 어떻게 된다고 했지?"}, config={"configurable": {"session_id": "test_v1"}})

result1 = conversational_rag_chain.invoke(
    {
        "input": "30인 고용인이 있는 자영업자야 내가 중대재해처벌법에 적용되는지 알고싶어"
    },
    config={"configurable": {"session_id": "test_v1"}}
)
result1