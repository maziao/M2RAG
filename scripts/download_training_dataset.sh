mkdir -p ./data/training_set
mkdir -p ./data/images/ELI5/training_set

huggingface-cli download ylwt/M2RAG-Distill-GPT-4o \
  --repo-type dataset \
  --local-dir ./data/training_set

for i in $(seq 0 9);
do
  echo "operating on training_set-${i}.tar"
  tar -xf ./data/training_set/images/ELI5/training_set/training_set-${i}.tar --directory=./data/images/ELI5/training_set
done

echo "operating on data.tar.gz"
tar -zxf ./data/training_set/data.tar.gz --directory=./data/training_set
