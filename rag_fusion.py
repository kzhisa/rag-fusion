"""
RAG Fusion sample
"""
import argparse
import os
import sys
from operator import itemgetter

from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

# LLM model
LLM_MODEL_OPENAI = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', help='Query with RRF search')
parser.add_argument('-r', '--retriever', help='Retrieve with RRF retriever')
parser.add_argument('-v', '--vector', help='Query with vector search')

# Retriever options
TOP_K = 5
MAX_DOCS_FOR_CONTEXT = 8
DOCUMENT_URL = "https://ja.wikipedia.org/wiki/%E5%8C%97%E9%99%B8%E6%96%B0%E5%B9%B9%E7%B7%9A"

# .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Template
my_template_jp = """Please answer the [question] using only the following [information] in Japanese. If there is no [information] available to answer the question, do not force an answer.

Information: {context}

Question: {question}
Final answer:"""



def load_and_split_document(url: str) -> list[Document]:
    """Load and split document

    Args:
        url (str): Document URL

    Returns:
        list[Document]: splitted documents
    """

    # Read the Wep documents from 'url'
    raw_documents = WebBaseLoader(url).load()

    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    # Split the documents
    documents = text_splitter.split_documents(raw_documents)

    # for TEST
    print("Original document: ", len(documents), " docs")

    return documents


def create_retriever(search_type: str, kwargs: dict) -> BaseRetriever:
    """Create vector retriever

    Args:
        search_type (str): search type 
        kwargs (dict): kwargs

    Returns:
        BaseRetriever: Retriever
    """


    # load and split document
    documents = load_and_split_document(DOCUMENT_URL)

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents, embeddings)

    # retriever
    retriever = vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=kwargs,
    )

    return retriever



def reciprocal_rank_fusion(results: list[list], k=60):
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): retrieved documents
        k (int, optional): parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: list of documents reranked by RRF
    """

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # for TEST (print reranked documentsand scores)
    print("Reranked documents: ", len(reranked_results))
    for doc in reranked_results:
        print('---')
        print('Docs: ', ' '.join(doc[0].page_content[:100].split()))
        print('RRF score: ', doc[1])

    # return only documents
    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]



def query_generator(original_query: dict) -> list[str]:
    """Generate queries from original query

    Args:
        query (dict): original query

    Returns:
        list[str]: list of generated queries 
    """

    # original query
    query = original_query.get("query")

    # prompt for query generator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("user", "Generate multiple search queries related to:  {original_query}. When creating queries, please refine or add closely related contextual information in Japanese, without significantly altering the original query's meaning"),
        ("user", "OUTPUT (3 queries):")
    ])

    # LLM model
    model = ChatOpenAI(
                temperature=0,
                model_name=LLM_MODEL_OPENAI
            )

    # query generator chain
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # gererate queries
    queries = query_generator_chain.invoke({"original_query": query})

    # add original query
    queries.insert(0, "0. " + query)

    # for TEST
    print('Generated queries:\n', '\n'.join(queries))

    return queries


def rrf_retriever(query: str) -> list[Document]:
    """RRF retriever

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Retriever
    retriever = create_retriever(search_type="similarity", kwargs={"k": TOP_K})

    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # invoke
    result = chain.invoke({"query": query})

    return result



def query(query: str, retriever: BaseRetriever):
    """
    Query with vectordb
    """

    # model
    model = ChatOpenAI(
        temperature=0,
        model_name=LLM_MODEL_OPENAI)

    # prompt
    prompt = PromptTemplate(
        template=my_template_jp,
        input_variables=["context", "question"],
    )

    # Query chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt | model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )

    # execute chain
    result = chain.invoke({"question": query})

    return result


# main
def main():

    # OpenAI API KEY
    if os.environ.get("OPENAI_API_KEY") == "":
        print("`OPENAI_API_KEY` is not set", file=sys.stderr)
        sys.exit(1)

    # args
    args = parser.parse_args()

    # retriever
    retriever = RunnableLambda(rrf_retriever)

    # query
    if args.query:
        retriever = RunnableLambda(rrf_retriever)
        result = query(args.query, retriever)
    elif args.retriever:
        retriever = RunnableLambda(rrf_retriever)
        result = rrf_retriever(args.retriever)
        sys.exit(0)
    elif args.vector:
        retriever = create_retriever(
            search_type="similarity",
            kwargs={"k": MAX_DOCS_FOR_CONTEXT}
        )
        result = query(args.vector, retriever)
    else:
        sys.exit(0)

    # print answer
    print('---\nAnswer:')
    print(result['response'])


if __name__ == '__main__':
    main()
