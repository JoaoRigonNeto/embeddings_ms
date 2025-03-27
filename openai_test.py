from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="The cake is a lie")
models = [
# "all-MiniLM-L12-v2",
# "all-mpnet-base-v2",
# "paraphrase-multilingual-MiniLM-L12-v2",
"distiluse-base-multilingual-cased-v2",
"all-MiniLM-L6-v2",
# "paraphrase-xlm-r-multilingual-v1",
# "all-distilroberta-v1",
# "msmarco-distilbert-base-tas-b",
# "xlm-r-distilroberta-base-paraphrase-v1",
]

for model in models:
    print(client.embeddings.create(input="this is a test", model=model))
# print(client.embeddings.create(input=["test1", "test2"], model=model))
