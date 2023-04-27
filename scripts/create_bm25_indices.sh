python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input codebm25_indices/JuICe_train/code_topics \
  --index codebm25_indices/JuICe_train/code_topics \
  --generator DefaultLuceneDocumentGenerator \
  --threads 2 \
  --storeDocvectors --storeRaw \
  --keepStopwords \
  --stemmer none \
  --pretokenized
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input codebm25_indices/JuICe_train/code_asts \
  --index codebm25_indices/JuICe_train/code_asts \
  --generator DefaultLuceneDocumentGenerator \
  --threads 2 \
  --storeDocvectors --storeRaw \
  --keepStopwords \
  --stemmer none \
  --storePositions