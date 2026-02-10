import os
from mem0 import Memory
import shutil
def create_memory_instance(collection_name,id,vlm_model,from_stratch=True,memory_base="5min"):
    if os.path.exists(f"./memory/{memory_base}"):
        instance_dir = f"./memory/{memory_base}/instance/{id}"
        if from_stratch:
            shutil.rmtree(instance_dir,ignore_errors=True)
            os.makedirs(instance_dir,exist_ok=True)
            shutil.copytree(f"./memory/{memory_base}/initial_memory", f"{instance_dir}/faiss")
            shutil.copyfile(f"./memory/{memory_base}/history/initial_memory.db", f"{instance_dir}/initial_memory.db")
        else:
            if not os.path.exists(instance_dir):
                os.makedirs(instance_dir,exist_ok=True)
                shutil.copytree(f"./memory/{memory_base}/initial_memory", f"{instance_dir}/faiss")
                shutil.copyfile(f"./memory/{memory_base}/history/initial_memory.db", f"{instance_dir}/initial_memory.db")
    
        config = {
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "memory",
                    "path": f"{instance_dir}/faiss",
                    "distance_strategy": "euclidean"
                }
            },
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4.1-mini", "temperature": 0.1},
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large"
                }
            },
            "reranker": {
                "provider": "llm_reranker",
                "config": {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4.1-mini", 
                            "temperature": 0,
                            "max_tokens": 500
                        }
                    },
                    "batch_ranking": True,  # Rank multiple at once
                    "top_n": 5,  # Fewer results for faster processing
                    "timeout": 10  # Request timeout
                }
        },
        "history_db_path":f"{instance_dir}/initial_memory.db"
        }

        return Memory.from_config(config)
    else:
        config = {
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": "memory",
                    "path": f"./memory/{memory_base}/initial_memory",
                    "distance_strategy": "euclidean"
                }
            },
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4.1-mini", "temperature": 0.1},
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large"
                }
            },
            "reranker": {
                "provider": "llm_reranker",
                "config": {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4.1-mini", 
                            "temperature": 0,
                            "max_tokens": 500
                        }
                    },
                    "batch_ranking": True,  # Rank multiple at once
                    "top_n": 5,  # Fewer results for faster processing
                    "timeout": 10  # Request timeout
                }
        },
        "history_db_path":f"./memory/{memory_base}/history/initial_memory.db"
        }

        return Memory.from_config(config)

def delete_memory_files(id,memory_base="5min"):
    instance_dir = f"./memory/{memory_base}/instance/{id}"
    shutil.rmtree(instance_dir,ignore_errors=True)