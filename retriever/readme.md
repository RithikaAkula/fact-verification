## Build Lucene Indices for BM25 using the following steps:

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

## Build Faiss Indices for DPR using the following steps:

### To create indices from "docs":

#### Run the file fact-verification/preprocessing/get-wiki-docs.ipynb

#### From clean text:

python -m pyserini.encode \
  input --corpus preprocessing/pyserini_format_docs_clean_texts --shard-id 0 --shard-num 1 \
  output --embeddings retriever/dpr/doc_indices/from_clean_texts --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco --batch 32 --fp16

#### From raw text:

python -m pyserini.encode \
  input --corpus preprocessing/pyserini_format_docs_raw_texts --shard-id 0 --shard-num 1 \
  output --embeddings retriever/dpr/doc_indices/from_raw_texts --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco --batch 32 --fp16

### To create indices from "passages":

#### Run the file fact-verification/preprocessing/get-wiki-passages.ipynb

#### From clean text:

python -m pyserini.encode \
  input --corpus preprocessing/pyserini_format_passages_clean_texts --shard-id 0 --shard-num 1 \
  output --embeddings retriever/dpr/passage_indices/from_clean_texts --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco --batch 32 --fp16

#### From raw text:

python -m pyserini.encode \
  input --corpus preprocessing/pyserini_format_passages_raw_texts --shard-id 0 --shard-num 1 \
  output --embeddings retriever/dpr/passage_indices/from_raw_texts --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco --batch 32 --fp16
