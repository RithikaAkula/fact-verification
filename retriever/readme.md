## Build Lucene Indices for BM25 using the following steps:

1. Create a directory to save the indices in. In this case, the name of the directory is: bm25_indices

2. Run the following commands in the terminal from the fact verification directory:

### To create indices from "passages":

#### Run the file fact-verification/preprocessing/get-wiki-passages.ipynb

#### From clean text:

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input preprocessing/pyserini_format_passages_clean_texts \
  --index retrievers/bm25/passage_indices/from_clean_texts \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw

#### From raw text:

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input preprocessing/pyserini_format_passages_raw_texts \
  --index retrievers/bm25/passage_indices/from_raw_texts \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw

### To create indices from "docs":

#### Run the file fact-verification/preprocessing/get-wiki-docs.ipynb

#### From clean text:

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input preprocessing/pyserini_format_docs_clean_texts \
  --index retriever/bm25/doc_indices/from_clean_texts \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw


#### From raw text:

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input preprocessing/pyserini_format_docs_raw_texts \
  --index retrievers/bm25/doc_indices/from_raw_texts \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw