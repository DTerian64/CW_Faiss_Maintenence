import os
from azure.cosmos import CosmosClient, PartitionKey

from dotenv import load_dotenv


class CosmicWorksDb:
    def __init__(self, endpoint, key, database_name):

        load_dotenv()        
        self.client = CosmosClient(endpoint, credential=key)              
        self.database = self.client.get_database_client(database_name)                
        
        self.ai_conversations_container = self.client.get_database_client(os.getenv("AICCONVERSATIONS_DATABASE_NAME") ).get_container_client("AIConversations")

    def get_container(self, container_name):
        return self.database.get_container_client(container_name)

    def search_products(self, keyword: str)-> list:
        keyword = keyword.strip()
        print(f"Searching for products with keyword: {keyword}")
        
        if not keyword:
            query = f"SELECT * FROM products p"
        else:
            query = f"SELECT * FROM products p WHERE CONTAINS(LOWER(p.name), LOWER(@keyword))"
        
        items = list(self.get_container("products").query_items(
            query=query,
            parameters=[{"name": "@keyword", "value": keyword}],
            enable_cross_partition_query=True
        ))
        return items

    def search_employees(self, keyword: str)-> list:
        keyword = keyword.strip()
        print(f"Searching for employees with keyword: {keyword}")
        
        if not keyword:
            query = f"SELECT * FROM employees p"
        else:
            query = f"SELECT * FROM employees p WHERE CONTAINS(LOWER(p.name), LOWER(@keyword))"
        
        items = list(self.get_container("employees").query_items(
            query=query,
            parameters=[{"name": "@keyword", "value": keyword}],
            enable_cross_partition_query=True
        ))
        return items
    
    def create_conversation_item(self, body: dict):
        """
        Create a new conversation item in the AIConversations container.
        """
        try:
            return self.ai_conversations_container.create_item(body=body)
        except Exception as e:
            print(f"Error creating conversation item: {e}")
            raise