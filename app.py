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
from dataloader import VTTLoader
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent import *
import openai
index_name = "langchain-demo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """PDF Chat Demo"""
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def process_file(file: AskFileResponse):
    import tempfile
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    with open(file.path, "r", encoding="utf-8") as f:
        content = f.read()
        # loader = Loader(tempfile.name)
        # documents = loader.load()
        docs = text_splitter.split_documents(content)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs



def loadRealData():
    #file_paths = ["./transcripts/controlling.vtt", "./transcripts/management.vtt ","./transcripts/marketing.vtt"]
    file_paths = "./transcripts"
    loader = VTTLoader(file_path=file_paths)
    documents = loader.load()

    #print(documents)
    #print(len(documents))
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore
'''
    # query it
    query = "Wann ist die Messe?"
    docs = vectorstore.similarity_search(query)

    # print results
    print(docs[0])
    '''
def generate_chatgpt_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",  # or other engine like "gpt-3.5-turbo"
          messages=[
    {"role": "user", "content": prompt}
  ]
    )
    
    return response
    
@cl.on_chat_start
async def start():
    vectorstore = loadRealData()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    topic = await cl.AskUserMessage(content=start_message).send()
    context = topic['output']
    #context += '. Beantworten die beigefügten Dokumente die genannte Frage vor diesem Satz? Antworte ausschliesslich mit Ja oder Nein'
    res = await chain.ainvoke(context, callbacks=[cb])
    # Example prompt
    prompt = "Hier ist das relevante Dokument: " + res['source_documents'][0].page_content + ". " + "Enthält dieses Dokument informationen über das folgende thema: " +context + ". Antworte nur mit Ja oder Nein"  

    # Generate response
    response = generate_chatgpt_response(prompt)
    response = response.choices[0].message.content


    if response == 'Ja':
        #source_name = f"source_{source_idx}"
        text = [cl.Text(content=res['source_documents'][0].page_content, name="summary")]

        await cl.Message(content='Du darfst kein Meeting erstellen. Es gibt Informationen', elements=text).send()
    if response == 'Nein':
        await cl.Message(content='Du darfst ein Meeting erstellen. Ich mache das für dich').send()



    #docs = vectorstore.similarity_search(query['output'])



@cl.on_message
async def main(message: cl.Message):

    # TODO now we have to determine if meeting topic has been handled in previous meetings

    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
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