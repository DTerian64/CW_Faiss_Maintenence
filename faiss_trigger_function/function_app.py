import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add your project paths
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

import azure.functions as func
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# Adjust this import path based on your project structure
from Agent.CosmicWorksLangChain import CosmicWorksLangChain

# Create the function app - this is required for v2 model
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.cosmos_db_trigger(
    arg_name="documents", 
    database_name="cosmicworks",
    container_name="products",
    connection="CosmosDB",  # This must match your local.settings.json key
    lease_container_name="leases",
    create_lease_container_if_not_exists=True
)
    
def update_index(documents: func.DocumentList) -> None:
    """
    Cosmos DB trigger function that updates FAISS index when documents change
    """
    # Uncomment these lines to enable debugging
    # import debugpy
    # if not debugpy.is_client_connected():
    #     debugpy.listen(('127.0.0.1', 5678))
    #     logging.info("🐛 Debugger listening on port 5678. Attach your debugger now!")
    #     debugpy.wait_for_client()
    
    logging.info("🔥 Function update_index triggered!")
    
    if documents:
        logging.info(f"📄 Processing {len(documents)} document(s) from products container.")
        
        # Log each document for debugging
        for i, doc in enumerate(documents):
            logging.info(f"Document {i}: {doc}")

        try:
            logging.info("🔧 Initializing Azure OpenAI models...")
            
            logging.info("📊 Creating FAISS index...")

            # Process the documents
            cosmic_lang = CosmicWorksLangChain()
            cosmic_lang.create_local_faiss_index(container_name="products")
            
            logging.info("☁️ Uploading to blob storage...")
            cosmic_lang.upload_faiss_index_to_blob(local_path="faiss_cosmicworks")

            logging.info("✅ FAISS index updated successfully.")
            
        except Exception as e:
            logging.error(f"❌ Error updating FAISS index: {str(e)}")
            logging.exception("Full error details:")  # This shows the full stack trace
            raise  # Re-raise to mark function as failed
    else:
        logging.info("ℹ️ No documents to process.")