python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input JuICe_train_code_cells_index \
  --index JuICe_train_code_cells_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 2 \
  --storeDocvectors --storeRaw