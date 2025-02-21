mkdir -p ./data/dev_set
mkdir -p ./data/images/ELI5/dev_set

huggingface-cli download ylwt/M2RAG-Bench \
  --repo-type dataset \
  --local-dir ./data/dev_set

for i in $(seq 0 9);
do
  echo "operating on dev_set-${i}.tar"
  tar -xf ./data/dev_set/images/ELI5/dev_set/dev_set-${i}.tar --directory=./data/images/ELI5/dev_set
done

echo "operating on dev_set.jsonl.tar.gz"
tar -zxf ./data/dev_set/dev_set.jsonl.tar.gz --directory=./data/dev_set
