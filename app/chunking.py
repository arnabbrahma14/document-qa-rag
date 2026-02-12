def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]

        start = 0
        count = 1
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append({
                "page_number":count,
                "content": chunk
            })

            start += chunk_size - overlap
            count += 1

    return chunks
