from pymilvus import MilvusClient
from pymilvus.milvus_client import IndexParams


def create_database(name:str, drop_collection:bool, dim:int, index_param:IndexParams | None) -> MilvusClient | None:
    """
    Create a database in Milvus.
    :param index_param: Index parameters for the database.
    :param dim: Dimension of the vectors to be stored in the database.
    :param drop_collection: If True, drop the collection if it exists.
    :param name: Name of the database to create.
    :return: None
    """

    client = MilvusClient(f"{name}_milvus.db")

    if drop_collection and client.has_collection(name):
        print(f"Collection {name} already exists. Dropping it.")
        client.drop_collection(name)

    elif client.has_collection(name):
        print(f"Collection {name} already exists.")
        return client

    client.create_collection(
        collection_name=name,
        dimension=dim,
        index_params=index_param,
    )

    print(f"Collection {name} created.")

    return client