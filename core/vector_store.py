import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.s3_utils import upload_dir, download_dir

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_and_store_faiss(docs, bucket, doc_id, base_prefix):
    with tempfile.TemporaryDirectory() as tmp:
        faiss_index = FAISS.from_documents(docs, embeddings)
        faiss_index.save_local(tmp)

        upload_dir(
            tmp,
            bucket,
            f"{base_prefix}{doc_id}/"
        )

def load_faiss_from_s3(bucket, doc_ids, base_prefix):
    indexes = []

    for doc_id in doc_ids:
        tmp = tempfile.mkdtemp()
        download_dir(bucket, f"{base_prefix}{doc_id}/", tmp)
        indexes.append(FAISS.load_local(tmp, embeddings))

    # 🔥 MULTI-DOC MERGE
    base = indexes[0]
    for idx in indexes[1:]:
        base.merge_from(idx)

    return base
