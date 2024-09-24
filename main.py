from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    prompt:str
class Response(BaseModel):
    response :str


qdrant_url="https://c60ed4c9-6696-4ba5-a0bf-c6825621deda.europe-west3-0.gcp.cloud.qdrant.io"
qdrant_api="xDbMjXd_oCdE3MkhMfbMBvjmYuMoLzP45in2jXpI9ElIt0hQqMFwNQ"
groq_api = "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"

# First Aid for the USMLE Step 1 2024 34th Edition [Medicalstudyzone.com].pdf
# [Medicalstudyzone.com] Pathoma 2023 PDF.pdf
def doc_loader():
    # loader=PyPDFLoader("C:\\Users\\Obaid Sajjad\\Desktop\\python\\First Aid for the USMLE Step 1 2024 34th Edition [Medicalstudyzone.com].pdf")
    # text = loader.load()
    book="C:\\Users\\Obaid Sajjad\\Desktop\\python\\First Aid for the USMLE Step 1 2024 34th Edition [Medicalstudyzone.com].pdf"
    text=""
    with open(book, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_chunking(doc):
    splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=900, 
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(doc)
    doc_list=[]
    for chunk in chunks:
        metadata={"source":"First Aid for the USMLE Step 1 2024 34th Edition"}
        document=Document(page_content=chunk, metadata=metadata)
        doc_list.append(document)
    return doc_list

def get_vectorstore(chunks,collection):
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base=Qdrant.from_documents(
        documents=chunks,
        collection_name=collection,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api,
        prefer_grpc=True
    )
    return knowledge_base

def get_knowledge_base(collection):
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base=QdrantVectorStore.from_existing_collection(
        collection_name=collection,
        embedding=embeddings,
        
        url=qdrant_url,
        api_key=qdrant_api,
    )
    return knowledge_base

def get_conversational_chain(vectorstore):
    llm = ChatGroq(model_name="Llama3-70b-8192", temperature=0.5, groq_api_key=groq_api)
    memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain



    # docs=doc_loader()
    # print("File Uploaded")
    # chunks = get_chunking(docs)
    # print("Chunks Created")
collection="Pathoma"
    # knowledge_base = get_vectorstore(chunks,collection)
    # print("Vector Store Created")

knowledge_base= get_knowledge_base(collection)
    # print(type(result))
    # print("Knowledge base Loaded")
@app.post('/prompt', response_model=Response)
async def getResponse(request:Item):
    # question="I have white dots on nails. diagnose me"
    prompt = request.prompt

    conversation_chain = get_conversational_chain(knowledge_base)
    system_role = "You are a helpful and knowledgeable medical assistant. Answer the following question based on medical knowledge.And try to diagnose patient disease and suggest him some medicines. If you have insufficient knowledge, say: consult with Doctor"
    result = conversation_chain({"question": f"{system_role} {prompt}"})

    return Response(response=result['answer'])
    # ans = result.similarity_search(question)
    # print(ans[0].page_content, "  ", ans[0].metadata)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.18.67", port=8000)
