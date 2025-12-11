from weaviate import Client
import numpy
import json


client = Client("http://localhost:8080")

schema = client.schema.get()
print(json.dumps(schema, indent=2))
results = client.query.get(
    "ArtEmbedding", 
    [
        "filename",
        "filepath",
        "style",
        "texture",
        "color",
        "emotion",
        "_additional {id, vector}"
    ]
).with_limit(3).do()
print(json.dumps(results, indent=2))