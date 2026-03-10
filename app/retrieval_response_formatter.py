def format_results(data, retriever):
    """
    Format retrieval results into a unified schema for the pipeline.

    Parameters
    ----------
    data : 
        - Pinecone result dict (for vector search)
        - List of chunks (for BM25)

    retriever : str
        "vector" or "bm25"

    scores : list or numpy array (optional)
        Required only for BM25
    """

    formatted_results = []

    if retriever == "vector":

        for match in data["matches"]:

            formatted_chunk = {
                "id": match["id"],
                "text": match["metadata"]["text"],
                "document_name": match["metadata"]["document_name"],
                "page": match["metadata"]["page_number"],
                "score": match["score"],
                "retriever": "vector"
            }

            formatted_results.append(formatted_chunk)

    elif retriever == "bm25":

        for chunk, score in data:

            formatted_chunk = {
                "id": chunk["id"],
                "text": chunk["text"],
                "document_name": chunk["document_name"],
                "page": chunk["page_number"],
                "score": float(score),
                "retriever": "bm25"
            }

            formatted_results.append(formatted_chunk)

    else:
        raise ValueError("Retriever must be either 'vector' or 'bm25'")

    return formatted_results