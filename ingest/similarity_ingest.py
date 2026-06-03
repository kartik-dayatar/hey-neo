from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance,PointStruct
import ollama
import uuid
import pathlib
import re
from ingest.bm25_ingest import apt_packages

def chunk_files(path:str) -> list[dict]:
    with open(path,"r") as file:
        content = file.readlines()

    text = []
    chunk = ""
    section = ""
    for line in content:

        if line.startswith("="*10):
            continue

        elif line.startswith("==="):
            if chunk:
                text.append({"text":chunk.strip(),"payload":{"source":pathlib.Path(path).stem,"content":chunk,"section":section}})
                chunk = ""
            section = line.split("===")[1].strip()

        elif not line.startswith("===") and line.strip():
            chunk +=line.strip()+" "
        
    text.append({"text":chunk.strip(),"payload":{"source":pathlib.Path(path).stem,"content":chunk,"section":section}})

    return text



_noice = re.compile(r'[:<>@,\[\]{}#]')
def split_into_chunks(text: str, max_words: int = 150, overlap: int = 35) -> list[str]:
    words = [w for w in text.split() 
    if not w.startswith('/') 
    and len(w)<=35 
    and not w.startswith("#")
    and not w.startswith("_")
    and not _noice.search(w)
    and any(c.isalpha() for c in w)
    ]  
    result = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        result.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words-overlap
    return result

texts = []
lines = chunk_files("./docs/hardware_info.txt")
lines.extend(chunk_files("./docs/system_info.txt"))
lines.extend(chunk_files("./docs/services.txt"))
lines.extend(chunk_files("./docs/network_info.txt"))
lines.extend(chunk_files("./docs/docker_info.txt"))
lines.extend(chunk_files("./docs/ollama_info.txt"))
packages = apt_packages("./docs/apt_packages.txt")


for line in lines:
    chunk = split_into_chunks(line['text'])
    for i in chunk:
        texts.append({
            "text":i,
            "payload":line['payload']
        })

qdrant = QdrantClient(url='http://localhost:6333')
if not qdrant.collection_exists("Similarity_search") and not qdrant.collection_exists("packages"):
    qdrant.create_collection("Similarity_search",
        vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))
    qdrant.create_collection('packages',
    vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))
else:
    qdrant.delete_collection("Similarity_search")
    qdrant.create_collection("Similarity_search",
        vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))
    qdrant.delete_collection('packages')
    qdrant.create_collection('packages',
    vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))

vector = []
for i in texts:
    print(f"words: {len(i['text'].split())}, source: {i['payload']['source']}, section: {i['payload']['section']}")
    response =  ollama.embeddings(model="mxbai-embed-large:latest",prompt=i['text'])
    vector.append(response['embedding'])

pack_vec = []
for i in packages:
    print(f"words: {len(i['text'].split())}, source: {i['payload']['source']}")
    response =  ollama.embeddings(model="mxbai-embed-large:latest",prompt=i['text'])
    pack_vec.append(response['embedding'])

for vect,text in zip(vector,texts):
    qdrant.upsert(collection_name="Similarity_search",
    points=[PointStruct(
        id=uuid.uuid4(),
        vector=vect,
        payload=text['payload']
    )
    ],
    wait=True
    )

for vect,pkg in zip(pack_vec,packages):
    qdrant.upsert(collection_name='packages',
    points=[PointStruct(
        id=uuid.uuid4(),
        vector=vect,
        payload=pkg['payload']
    )
    ],
    wait=True
    )
    