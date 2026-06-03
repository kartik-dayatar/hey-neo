from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

    
#reading all files and creating chunks acording to format of data
def apt_packages(path:str) -> list[dict]:

    with open (path, "r") as f:
        content = f.readlines()

    packages = [package.strip() for package in content if package.startswith('ii')]

    tokens = [item.split() for item in packages]

    chunks = []

    for token in tokens:
        chunk={
            'text':token[1].lower()+" "+" ".join(token[4:]).lower(),
            "payload":{
                "name":token[1].lower(),
                "version":token[2].lower(),
                "desc":" ".join(token[4:]).lower(),
                "source":"apt_packages"
            }
        }
        chunks.append(chunk)
    return chunks

def apt_manual(path:str) -> list[dict]:

    with open (path, "r") as f:
        content = f.readlines()
    packages = [package.strip() for package in content[1:] if package.strip()]
    chunks = []

    for token in packages:
        chunk={
            'text':token.lower(),
            "payload":{
                "name":token.lower(),
                "source":"apt_manual"
            }
        }
        chunks.append(chunk)
    return chunks

def python_package(path:str) -> list[dict]:
    with open (path, "r") as f:
        content = f.readlines()
    packages = [package.split() for package in content[3:] if package.strip() and len(package.split()) >= 2]
    chunks = []

    for token in packages:
        chunk={
            'text':token[0].lower(),
            "payload":{
                "name":token[0].lower(),
                "version":token[1],
                "source":"python_packages"
            }
        }
        chunks.append(chunk)
    return chunks

def flatpak_package(path:str) -> list[dict]:
    with open (path, "r") as f:
        content = f.readlines()
    packages = [package.split('\t') for package in content[1:] if package.strip()]
    chunks = []

    for token in packages:
        chunk={
            'text':token[0].lower(),
            "payload":{
                "name":token[0].lower(),
                "appID":token[1].lower(),
                "version":token[2].lower(),
                "info":token[3:],
                "source":"flatpak_package"
            }
        }
        chunks.append(chunk)
    return chunks

def conda_info(path:str) -> list[dict]:
    with open(path, "r") as f:
        content = f.readlines()

    chunks = []

    for line in content:
        raw_line = line.rstrip("\n")
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("==="):
            continue

        if stripped.startswith("#"):
            continue

        is_continuation = raw_line and raw_line[0] in (" ", "\t") and " : " not in stripped

        if is_continuation and chunks:
            last = chunks[-1]
            if last["payload"].get("source") == "conda_sysinfo":
                prev_value = last["payload"]["value"]
                if isinstance(prev_value, list):
                    prev_value.append(stripped)
                else:
                    last["payload"]["value"] = [prev_value, stripped]
            continue  

        if " : " in stripped:
            parts = stripped.split(" : ", 1)
            key = parts[0].strip()
            value = parts[1].strip()
            chunks.append({
                "text": key,
                "payload": {"key": key, "value": value, "source": "conda_sysinfo"}
            })

        elif stripped.startswith("Python"):
            chunks.append({
                "text": stripped,
                "payload": {"version": stripped, "source": "python_versions"}
            })

        elif stripped[0].isalpha():
            parts = stripped.split()
            name = parts[0]
            path_v = parts[-1]
            chunks.append({
                "text": f"conda environment {name}",
                "payload": {"name": name, "path": path_v, "source": "conda_envs"}
            })

        elif stripped.startswith("/"):
            chunks.append({
                "text": f"python path {stripped}",
                "payload": {"path": stripped, "source": "python_versions"}
            })
    return chunks

def shell_config(path:str) -> dict:
    with open(path,'r') as file:
        content = file.readlines()

    text = []

    chunk = {
        "text":"alias",
        "payload":{}
    }
    temp_config=[]
    for line in content:
        if line.startswith("alias"):
            word = line.split("=",1)
            alias = word[0][6:]
            conf = word[1]
            temp_config.append(conf)
            text.append(alias)
        else:
            continue

    chunk["payload"] = {"alias":text,"config":temp_config,"source":"shell_config"}
    chunk["text"] = " ".join(text)
    return chunk

#combining all chunks which are under BM25 serch
chunks = []
chunks.extend(apt_packages("./docs/apt_packages.txt"))
chunks.extend(apt_manual("./docs/apt_manual.txt"))
chunks.extend(python_package("./docs/python_packages.txt"))
chunks.extend(flatpak_package("./docs/flatpak_packages.txt"))
chunks.extend(conda_info("./docs/conda_info.txt"))
chunks.append(shell_config("./docs/shell_config.txt"))

#building corpus for BM25
tokenized_corpus = [chunk['text'].split(" ") for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# --------------------For manual vocab management------------------------

# all_tokens = list(set(t for doc in tokenized_corpus for t in doc))
# vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}

# sparce_vec=[]

# for doc in tokenized_corpus:
#     unique_set = set(doc)
#     indice = [vocab[token] for token in unique_set]
#     values = [bm25.idf[token] for token in unique_set]
#     doc1 = (indice,values)
#     sparce_vec.append(doc1)


qdrant = QdrantClient(url='http://localhost:6333')
if not qdrant.collection_exists("BM25_search"):
    qdrant.create_collection("BM25_search",sparse_vectors_config={'bm25':models.SparseVectorParams()},on_disk_payload=True)



corpus = [chunk['text'] for chunk in chunks]
sparse_emabeder = SparseTextEmbedding(model_name="Qdrant/bm25")

sparse_vec = list(sparse_emabeder.embed(corpus))


qdrant.upsert(
    collection_name="BM25_search",
    points=[models.PointStruct(id=i, vector={"bm25":models.SparseVector(indices=sparse_vec[i].indices.tolist(), values=sparse_vec[i].values.tolist())}, payload=chunks[i]['payload']) for i, _ in enumerate(sparse_vec)]
)

