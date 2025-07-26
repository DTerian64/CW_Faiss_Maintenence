from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains.qa_with_sources.map_reduce_prompt import QUESTION_PROMPT
from langchain.chains.llm import LLMChain



from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
import os
import json
from azure.storage.blob import BlobServiceClient
from Helpers.MyCosmosDBHelper import CosmicWorksDb


class CosmicWorksLangChain:
    def __init__(self, chat_model: AzureChatOpenAI = None, embeddings: AzureOpenAIEmbeddings = None):
        
        self.chat_model = chat_model or AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),  # Your deployed chat model name
            openai_api_version="2023-05-15",
            temperature=0.7
        )
        self.embeddings = embeddings or AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g. https://your-resource-name.openai.azure.com
            deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),  # Your deployed embedding model name
            openai_api_version="2023-05-15",
            chunk_size=100 
        )
    
  
    def create_local_faiss_index(self, container_name: str):
        """Create a local FAISS index from Cosmos DB."""
        
        try:
            cosmos_db = CosmicWorksDb(
                endpoint=os.getenv("COSMOS_ENDPOINT"),
                key=os.getenv("COSMOS_PRIMARY_KEY"),
                database_name=os.getenv("COSMICWORKS_DATABASE_NAME")
            )
            
            items = list(cosmos_db.get_container(container_name).read_all_items())
            documents = [
                Document(
                    page_content=(
                        f"{item['name']}. {item['description']} "
                        f"Category: {item.get('category', {}).get('name', 'Unknown')} → "
                        f"SubCategory: {item.get('category', {}).get('subCategory', {}).get('name', 'Unknown')} "
                        f"Price: ${item['price']}"
                    ),
                    metadata={
                    "source": "cosmicworks",
                    "category": item.get('category', {}).get('name'),
                    "subCategory": item.get('category', {}).get('subCategory', {}).get('name')
                    }
                )
                for item in items
            ]

            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            
            vector_store = FAISS.from_documents(docs, self.embeddings)
            vector_store.save_local("faiss_cosmicworks")
            print("FAISS index created and saved localy.")
        except Exception as e:
            print(f"Error creating remote FAISS index: {e}")

    def upload_faiss_index_to_blob(self, local_path="faiss_cosmicworks"):
        conn_str = os.getenv("BLOB_CONNECTION_STRING")
        container = os.getenv("BLOB_CONTAINER_NAME")  # e.g. "vector-index"
        blob_dir = os.getenv("AZURE_FAISS_BLOB_DIR", "faiss_cosmicworks")

        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service.get_container_client(container)

        for fname in ["index.faiss", "index.pkl"]:
            file_path = f"{local_path}/{fname}"
            blob_name = f"{blob_dir}/{fname}"
            blob_client = container_client.get_blob_client(blob_name)

            with open(file_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)

    def download_faiss_index_from_blob(self,local_path="faiss_cosmicworks"):
        print("⏬ Downloading FAISS index from Azure Blob Storage...")

        connection_string = os.getenv("BLOB_CONNECTION_STRING")
        container_name = os.getenv("BLOB_CONTAINER_NAME")
        blob_dir = os.getenv("AZURE_FAISS_BLOB_DIR", "faiss_cosmicworks")

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)

        os.makedirs(local_path, exist_ok=True)
    
        for filename in ["index.faiss", "index.pkl"]:
            blob_name = f"{blob_dir}/{filename}"
            print(f"Downloading {blob_name} to {local_path}...")
            blob_client = container_client.get_blob_client(blob_name)
            download_path = os.path.join(local_path, filename)

            with open(download_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
    
    def faiss_index_exists(self, local_path="faiss_cosmicworks"):
        """Check if the FAISS index exists locally."""
        return (
            os.path.exists(os.path.join(local_path, "index.faiss")) and
            os.path.exists(os.path.join(local_path, "index.pkl"))
    )


    def get_process_langchain(self, user_input: str) -> str:
        """Answer a user_input using the RetrievalQA agent."""

        try:
            local_faiss_path = "faiss_cosmicworks"

            if not self.faiss_index_exists(local_faiss_path):
                self.download_faiss_index_from_blob(local_faiss_path)                      

            # Load vector store
            vector_store = FAISS.load_local(local_faiss_path, self.embeddings, allow_dangerous_deserialization=True)

            total_docs = vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0
            print(f"Total documents in vector store: {total_docs}")  

            # DYNAMIC K BASED ON QUERY TYPE
            counting_keywords = ["how many", "count", "number of", "total", "all"]
            is_counting_query = any(keyword in user_input.lower() for keyword in counting_keywords)
            
            k = 300 #min(200, total_docs)            
            
            # Updated prompt for better counting
            if is_counting_query:
                template = (
                "You are a helpful assistant for Cosmic Works product analysis.\n"
                "For counting questions, carefully count ALL items mentioned in the context.\n"
                "Be thorough and systematic in your counting.\n"
                "If the context seems incomplete, mention this limitation.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Count carefully and show your work:"
                )
            else:
                template = (
                    "You are a helpful assistant for product questions in Cosmic Works.\n"
                    "Given the following context, answer the request.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n"
                    )
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            ) 

            retriever=vector_store.as_retriever(search_kwargs={"k": k})        
            
            # Create the RetrievalQA with map_reduce chain
            #qa_chain = RetrievalQA.from_llm(
            #    llm=self.chat_model,
            #    retriever=retriever,
            #    return_source_documents=False,
            #    prompt=prompt
            #)
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.chat_model,
                retriever=retriever,
                chain_type="stuff",  # or "refine"
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
)

            print("✅ RetrievalQA agent initialized successfully.")

            response = qa_chain.invoke({"query": user_input})
            #answer = response["result"]
            #sources = response["source_documents"]


            #print(f"Answer: {response["result"]}")
            #print(f"Based on {len(sources)} source documents")
            #for i, doc in enumerate(sources):
            #    print(f"Source {i+1}: {doc.page_content[:100]}...")
                        
            return response["result"]

        except Exception as e:
            print(f"Error in get_process_langchain: {e}")
            return f"Error in get_process_langchain: {e}"
