from lazyllm import Document, Retriever
doc = Document(dataset_path='/home/mnt/lixin/firstrag/LazyRag/data')
seperator = '\n' + '='*200 + '\n'
retriever = Retriever(
    doc=doc,
    group_name="FineChunk",
    similarity="bm25_chinese",
    topk=3,
    output_format="content",
    join=seperator
)
print(retriever("出租管理"))