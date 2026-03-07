from rank_bm25 import BM25Okapi
# from chunking import chunk_text
# from ingestion import extract_text

class BM25Index:

    def __init__(self, chunks):
        """
        chunks: list of dictionaries containing chunk metadata
        """
        self.chunks = chunks

        # Extract text
        self.corpus = [chunk["content"] for chunk in chunks]

        # Tokenize
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=3):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
    
# pages = extract_text("../data/merchant_of_venice_summary.pdf")
# chunks = chunk_text(pages)
# bm_obj = BM25Index(chunks)

# #print("#" * 80 + "\n"+ "Token Corpus" + "\n" + "#" * 80 + "\n")
# for (x,y) in zip(bm_obj.corpus[:5], bm_obj.tokenized_corpus[:5]):
#     print(x)
#     print('\n' * 3)
#     print(y)
#     print("*" * 100)
    
# print("\n" * 3 + "TC")
# print(bm_obj.tokenized_corpus[:2])