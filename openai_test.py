from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="The cake is a lie")
model = "all-MiniLM-L6-v2"

print(client.embeddings.create(input="this is a test", model=model))
print(client.embeddings.create(input=["test1", "test2"], model=model))