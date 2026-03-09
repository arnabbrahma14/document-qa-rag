from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages, document_name):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    chunk_id = 0

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        split_chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(split_chunks):

            chunks.append({
                "id": f"{document_name}_chunk_{chunk_id}",
                "content": chunk,
                "metadata": {
                    "document_id": document_name,
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "source_type": "pdf"
                }
            })

            chunk_id += 1

    return chunks
