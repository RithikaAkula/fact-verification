## Build Lucene Indices for BM25 using the following steps:

1. Create a directory to save the indices in. In this case, the name of the directory is: bm25_indices

2. Run the following command from the terminal:

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input preprocessing/pyserini_format_passages_clean_texts \
  --index retrievers/bm25/doc_indices/from_clean_texts \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw
