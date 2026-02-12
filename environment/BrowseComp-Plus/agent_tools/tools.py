"""BrowseComp-Plus Tools - HTTP Client for FAISS Search Server

LangChain-compatible tools for the BrowseComp-Plus deep research benchmark.
Connects to a separate FAISS server to avoid segfaults in the main process.

Before using these tools, start the FAISS server:
    cd environment/BrowseComp-Plus/dataset
    python faiss_server.py
"""

from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

# Server configuration (fixed port)
FAISS_SERVER_URL = "http://127.0.0.1:8765"


def _check_server():
    """Check if FAISS server is running."""
    try:
        response = requests.get(f"{FAISS_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def _search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search documents via FAISS server."""
    try:
        response = requests.post(
            f"{FAISS_SERVER_URL}/search",
            json={"query": query, "k": k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return [{"error": "FAISS server not running. Start it with: python environment/BrowseComp-Plus/dataset/faiss_server.py"}]
    except Exception as e:
        return [{"error": str(e)}]


def _get_document_by_id(docid: str) -> Optional[Dict[str, Any]]:
    """Get document via FAISS server."""
    try:
        response = requests.get(
            f"{FAISS_SERVER_URL}/document/{docid}",
            timeout=30
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "FAISS server not running. Start it with: python environment/BrowseComp-Plus/dataset/faiss_server.py"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# LangChain Tools
# =============================================================================

@tool("search")
def search(query: str) -> List[Dict[str, Any]]:
    """Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits).

    Args:
        query: Search query string

    Returns:
        List of search results with docid, score, and snippet
    """
    return _search_documents(query, k=5)


@tool("get_document")
def get_document(docid: str) -> Optional[Dict[str, Any]]:
    """Retrieve a full document by its docid.

    Args:
        docid: Document ID to retrieve

    Returns:
        Document with full text, or None if not found
    """
    return _get_document_by_id(docid)


# Tool collections
ALL_TOOLS = [search, get_document]
TOOL_MAPPING = {"search": search, "get_document": get_document}


# Quick test
if __name__ == "__main__":
    print("Testing FAISS server connection...")
    if _check_server():
        print("Server is running!")
        print("\nTesting search...")
        results = search.invoke({"query": "test query"})
        print(f"Search results: {len(results)} items")
        if results and "error" not in results[0]:
            print(f"First result: {results[0]['docid']}")
    else:
        print("Server is NOT running.")
        print("Start it with: python environment/BrowseComp-Plus/dataset/faiss_server.py")
