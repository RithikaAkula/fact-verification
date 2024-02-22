import json
import os
from tqdm import tqdm

TOPK = 10


def perform_retrieval(data, use_raw_texts, use_doc_indices, base_dir):
    
    from pyserini.search.lucene import LuceneSearcher

    print("loading bm25 doc indices..." if use_doc_indices else "loading bm25 passage indices...")
    
    if use_doc_indices:
        parent_dir = f"{base_dir}/doc_indices"
        searcher = LuceneSearcher(f'{parent_dir}/from_raw_texts' if use_raw_texts else f'{parent_dir}/from_clean_texts')
    else:
        parent_dir = f"{base_dir}/passage_indices"
        searcher = LuceneSearcher(f'{parent_dir}/from_raw_texts' if use_raw_texts else f'{parent_dir}/from_clean_texts')
   
    # TO FINE-TUNE:
    '''
        k1 : float 
            BM25 k1 parameter.
            optimal range: (0.3, 0.9)
        b : float
            BM25 b parameter.
            range allowed: (0, 1)

        general values used: (1.5, 0.75)
        - on clean texts: (0.2, 0.05) - 0.03839
        - raw texts: (0.001, 0.5)
    '''
    searcher.set_bm25(0.001, 0.5)

    print("running bm25 retrieval...")

    top_k_docs = dict()
    top_k_docs["claims"] = []

    for d in tqdm(data):
        op = dict()
        query = d["raw_text"] if use_raw_texts else d["clean_text"]
        op["claim"] = query
        op["actual_ids"] = d["joint_ids"]
        hits = searcher.search(query, TOPK)
        
        docs = []
        for i in range(len(hits)):
            doc = searcher.doc(str(hits[i].docid))
            doc_contents = json.loads(doc.raw())
            if use_doc_indices:
                docs.append({
                    "rank": i+1,
                    "predicted_doc": doc_contents["contents"],
                    "doc_id": doc.id(),
                    "score": f"{hits[i].score:.5f}",
                })
            else:
                ids = doc.id().split("_")
                docs.append({
                    "rank": i+1,
                    "predicted_passage": doc_contents["contents"],
                    "predicted_passage_id": ids[1],
                    "doc_id": ids[0],
                    "joint_id": doc.id(),
                    "score": f"{hits[i].score:.5f}",
                })
        op["docs"] = docs
        top_k_docs["claims"].append(op)
        del op
    
    return top_k_docs

def extract_actual_predicted_ids(data, use_doc_indices):
    actual = []
    predicted = []

    # Iterate over claims
    for claim_data in data["claims"]:
        
        # ignore NEI rows
        if claim_data['actual_ids'] == []:
            continue

        # Extract the ground truth doc_ids list
        actual_ids = claim_data["actual_ids"]
        if use_doc_indices:
            actual_ids = [value.split('_')[0] for value in actual_ids]
        
        actual.append(actual_ids)

        # Extract predicted_ids while preserving order based on rank
        if use_doc_indices:
            predicted_ids = [doc["doc_id"] for doc in sorted(claim_data["docs"], key=lambda x: x["rank"])]
        else:
            predicted_ids = [doc["joint_id"] for doc in sorted(claim_data["docs"], key=lambda x: x["rank"])]
        predicted.append(predicted_ids)

    return actual, predicted


def precision_at_k(actual, predicted, k=10):
     scores = []
     for i in range(len(actual)):
         top_k_pred = predicted[i][:k]
         act_set = set(actual[i])
         pred_set = set(top_k_pred)
         result = len(act_set & pred_set) / len(pred_set)
         scores.append(result)
     return sum(scores) / len(scores)


def reciprocal_rank_at_k(actual, predicted, k=10):
    mrrs = []
    for i in range(len(actual)):
        if actual[i] == []:
            continue
        ranks = []
        seen = set()  # Keep track of seen relevant documents
        for j in range(len(actual[i])):
            if actual[i][j] not in seen:
                top_k_pred = predicted[i][:k]
                try:
                    rank = 1 / (top_k_pred.index(actual[i][j]) + 1)
                except ValueError:
                    rank = 0  # If actual[i][j] not in top_k_pred, set rank to 0
                ranks.append(rank)
                seen.add(actual[i][j])  # Mark this relevant document as seen
        # If no relevant document found, add 0 to MRRs
        if not ranks:
            mrrs.append(0)
        else:
            mrrs.append(sum(ranks) / len(ranks))
    return sum(mrrs) / len(mrrs)


def hits_ratio_at_k(actual, predicted, k=10):
    hits = 0
    for i in range(len(actual)):
        top_k_pred = predicted[i][:k]
        # Check if any relevant document is among the top k predicted documents
        if any(doc in actual[i] for doc in top_k_pred):
            hits += 1
    return hits / len(actual)


if __name__ == "__main__":
    
    use_doc_indices = True
    use_raw_texts = False
    on_complete_fever = False
    is_bm25 = True

    base_dir = 'bm25' if is_bm25 else 'dpr'

    if use_doc_indices:
        print("Retrieving from raw docs" if use_raw_texts else "Retrieving from clean docs")
    else:
        print("Retrieving from raw passages" if use_raw_texts else "Retrieving from clean passages")

    # CREATE DIRECTORY TO STORE RESULTS
    if use_doc_indices:
        parent_dir = f"{base_dir}/retrieved_docs"
        os.makedirs(parent_dir, exist_ok=True)  # Create the output directory if it doesn't exist

        if on_complete_fever:
            output_file_path = f"{parent_dir}/top_{TOPK}_raw_texts.json" if use_raw_texts else f"{parent_dir}/top_{TOPK}_clean_texts.json"
        else:
            output_file_path = f"{parent_dir}/top_{TOPK}_raw_texts_1000.json" if use_raw_texts else f"{parent_dir}/top_{TOPK}_clean_texts_1000.json"
    else:
        parent_dir = f"{base_dir}/retrieved_passages"
        os.makedirs(parent_dir, exist_ok=True)  # Create the output directory if it doesn't exist

        if on_complete_fever:
            output_file_path = f"{parent_dir}/top_{TOPK}_raw_texts.json" if use_raw_texts else f"{parent_dir}/top_{TOPK}_clean_texts.json"
        else:
            output_file_path = f"{parent_dir}/top_{TOPK}_raw_texts_1000.json" if use_raw_texts else f"{parent_dir}/top_{TOPK}_clean_texts_1000.json"
    
    # READ FEVER DATASET
    dirs = os.path.abspath(os.curdir).split("/")
    curr_path = "/".join(dirs[:-1])
    input_data_path = curr_path + ('/preprocessing/processed_fever/fever-train-split.json' if on_complete_fever else '/preprocessing/processed_fever/fever-train-1000.json')
    with open(input_data_path) as f:
        data = json.load(f)

    # PERFORM RETRIEVAL
    top_k_docs = perform_retrieval(data, use_raw_texts, use_doc_indices, base_dir)

    # STORE RETRIEVAL RESULTS
    with open(output_file_path, 'w') as json_file:
        json.dump(top_k_docs, json_file)
    
    # EVALUATE PERFORMANCE
    actual, predicted = extract_actual_predicted_ids(top_k_docs, use_doc_indices)
    
    print(f"MRR@{str(10)}:", reciprocal_rank_at_k(actual, predicted, 10))
    print(f"MRR@{str(5)}:", reciprocal_rank_at_k(actual, predicted, 5))
    print(f"MRR@{str(2)}:", reciprocal_rank_at_k(actual, predicted, 2))
    print()
    print(f"Hits@{str(10)}:", hits_ratio_at_k(actual, predicted, 10))
    print(f"Hits@{str(5)}:", hits_ratio_at_k(actual, predicted, 5))
    print(f"Hits@{str(2)}:", hits_ratio_at_k(actual, predicted, 2))
    print()
    print(f"Precision@{str(10)}:", precision_at_k(actual, predicted, 10))
    print(f"Precision@{str(5)}:", precision_at_k(actual, predicted, 5))
    print(f"Precision@{str(2)}:", precision_at_k(actual, predicted, 2))

