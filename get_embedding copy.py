from weaviate import Client
import json

client = Client("http://localhost:8080")

# Print schema
schema = client.schema.get()
print(json.dumps(schema, indent=2))

# Query vectors
results = (
    client.query
    .get(
        "ArtEmbedding", 
        [
            "filename",
            "filepath",
            "style",
            "texture",
            "color",
            "emotion",
            "_additional { uuid vector }"
        ]
    )
    .with_limit(3)
    .do()
)

print(json.dumps(results, indent=2))
