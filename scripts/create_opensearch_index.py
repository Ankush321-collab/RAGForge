import os
import sys
import argparse
from urllib.parse import urlparse
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Ensure backend package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.config import get_settings


def get_opensearch_client(host: str, username: str | None = None, password: str | None = None) -> OpenSearch:
    settings = get_settings()

    parsed = urlparse(host if host.startswith("http") else f"https://{host}")
    use_ssl = parsed.scheme == "https"
    port = parsed.port or (443 if use_ssl else 9200)
    hostname = parsed.hostname or host

    # Check if host is OpenSearch Serverless (.aoss.)
    is_aoss = ".aoss." in hostname

    if is_aoss:
        access_key = settings.s3_access_key_id
        secret_key = settings.s3_secret_access_key
        region = settings.s3_region or "ap-south-2"
        print(f"Detected OpenSearch Serverless (.aoss). Using AWS SigV4 Auth (Region: {region})...")
        auth = AWS4Auth(access_key, secret_key, region, "aoss")
    else:
        auth = (username, password) if username and password else None

    return OpenSearch(
        hosts=[{"host": hostname, "port": port}],
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=use_ssl,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
    )


def create_index(host: str, username: str | None, password: str | None, index_name: str, dimension: int = 384):
    print("==================================================")
    print("  AWS OpenSearch Vector Index Creator Script")
    print("==================================================")
    print(f"Target Host       : {host}")
    print(f"Target Index Name : {index_name}")
    print(f"Vector Dimensions : {dimension} (all-MiniLM-L6-v2)")

    client = get_opensearch_client(host, username, password)

    index_body = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "space_type": "cosinesimil"
                },
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "strategy": {"type": "keyword"}
            }
        }
    }

    try:
        # Check if index already exists
        if client.indices.exists(index=index_name):
            print(f"\nIndex '{index_name}' already exists in OpenSearch!")
            response = client.indices.get(index=index_name)
            print("Existing Index Mappings:")
            print(response)
            return

        # Create index
        print(f"\nCreating index '{index_name}'...")
        response = client.indices.create(index=index_name, body=index_body)
        print(f"Success! Response: {response}")

        # Confirm index mapping
        indices_info = client.indices.get_mapping(index=index_name)
        print(f"\nVerified Index '{index_name}' Mapping:")
        print(indices_info)
    except Exception as e:
        print(f"\nOpenSearch API Error: {e}")


def main():
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Create OpenSearch vector index for RAGForge.")
    parser.add_argument("--host", default=settings.opensearch_host, help="OpenSearch endpoint host")
    parser.add_argument("--username", default=settings.opensearch_username, help="OpenSearch master username")
    parser.add_argument("--password", default=settings.opensearch_password, help="OpenSearch master password")
    parser.add_argument("--index", default=settings.opensearch_index_name, help="Index name")
    parser.add_argument("--dimension", type=int, default=384, help="Vector dimension size")

    args = parser.parse_args()

    if not args.host:
        print("Error: OpenSearch host endpoint not provided.")
        print("Please specify --host or set OPENSEARCH_HOST in your .env file.")
        sys.exit(1)

    create_index(
        host=args.host,
        username=args.username,
        password=args.password,
        index_name=args.index,
        dimension=args.dimension,
    )


if __name__ == "__main__":
    main()
