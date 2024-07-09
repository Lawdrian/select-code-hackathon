from typing import List
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from agent import *

index_name = "langchain-demo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings(openai_api_base="https://llmproxy.meingpt.com")

welcome_message = """PDF Chat Demo"""


def process_file(file: AskFileResponse):
    import tempfile
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False) as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)
    cl.user_session.set("docs", docs)
    docsearch = Chroma.from_documents(docs, embeddings)
    return docsearch


@cl.on_chat_start
async def start():
    # files = None
    # while files is None:
    #     files = await cl.AskFileMessage(
    #         content=welcome_message,
    #         accept=["text/plain", "application/pdf"],
    #         max_size_mb=20,
    #         timeout=180,
    #     ).send()

    # file = files[0]

    msg = cl.Message(content=start_message)
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, base_url="https://llmproxy.meingpt.com"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()