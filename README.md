## RAG Fusion with RRF Sample

This repository provides a sample program for RAG Fusion using the Reciprocal Rank Fusion (RRF). RAG (Retrieval-Augmented Generation) techniques enhance tasks such as question answering and text generation by retrieving information from extensive text data and using it for generation. This sample specifically utilizes RRF to aim for more efficient information retrieval and integration.

### Prerequisites

To run this sample program, you'll need the following environment:

* Python 3.11 or newer
* All required Python libraries as listed in requirements.txt

To install these libraries, you'll need to use the following command:

```bash
pip install -r requirements.txt
```

### Setting DOCUMENT_URL

To specify a document URL for the program to load, set the DOCUMENT_URL variable before running the program at the line 34 of rag_fusion.py.


### How to Use

The sample program can be executed from the command line. To see the available options, use the command:

```
python rag_fusion.py --help
```

To retrieve documents from vector store, use the command:

```
python rag_fusion.py -r <query>
```

To answer the query using RAG Fusion retriever, use the command:

```
python rag_fusion.py -q <query>
```



