from lazyllm import Document, Retriever
import lazyllm


def first_rag():
    documents = Document("/home/mnt/lixin/firstrag/LazyRag/data")
    retriever = lazyllm.Retriever(
        doc=documents,
        group_name="CoarseChunk",
        similarity="bm25_chinese",
        topk=1,
        output_format="content",
        join=''
    )
    retriever.start()
    prompt = (
        'You will act as an AI question-answering assistant and complete a dialogue task.'  # noqa E501
        'In this task, you need to provide your answers based on the given context and questions.'  # noqa E501
    )
    llm = lazyllm.OnlineChatModule(source='sensenova').prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))  # noqa E501
    query = input("请输入您的问题：\n")
    res = llm({"query": query, "context_str": retriever(query=query)})
    print(f"With RAG Answer:\n{res}")


if __name__ == "__main__":
    first_rag()