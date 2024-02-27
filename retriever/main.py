import json
import os
from tqdm import tqdm
import pandas as pd

TOPK = 10


def extract_actual_predicted_ids(data, use_doc_indices):
    actual = []
    predicted = []

    # Iterate over claims
    for claim_id, claim_data in data.items():
        
        # Extract the ground truth doc_ids list
        actual_ids = claim_data["ground_truth_docs"]

        # ignore NEI rows
        if len(actual_ids) == 0:
            continue
        actual.append(actual_ids)

        # Extract predicted_ids while preserving order based on rank
        predicted_ids = claim_data['predicted_docs']
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


def dpr_retrieval(data, use_raw_texts, use_doc_indices, base_dir):

    print("loading faiss doc indices..." if use_doc_indices else "loading faiss passage indices...")
    
    if use_doc_indices:
        parent_dir = f"{base_dir}/doc_indices"
        indices_path = f'{parent_dir}/from_raw_texts' if use_raw_texts else f'{parent_dir}/from_clean_texts'
    else:
        parent_dir = f"{base_dir}/passage_indices"
        indices_path = f'{parent_dir}/from_raw_texts' if use_raw_texts else f'{parent_dir}/from_clean_texts'
    
    from pyserini.search import FaissSearcher
    searcher = FaissSearcher(indices_path, 'facebook/dpr-question_encoder-multiset-base')
    print("Running DPR retrieval...")

    claims = dict()

    for d in tqdm(data):
        op = dict() 
        query = d["raw_text"] if use_raw_texts else d["clean_text"]
        op["claim"] = query
        op["label"] = d["label"]
        ground_truth_doc_ids = list(d["joint_ids"])
        if use_doc_indices:
            ground_truth_doc_ids = list(set([value.split('_')[0] for value in ground_truth_doc_ids]))

        hits = searcher.search(query)
        
        predicted_doc_ids = []

        for i in range(len(hits)):
            predicted_doc_ids.append(str(hits[i].docid))
            
        op['ground_truth_docs'] = ground_truth_doc_ids
        op['predicted_docs'] = predicted_doc_ids
        
        claims[d["claim_id"]] = op 

        del op
    
    return claims


def bm25_retrieval(data, use_raw_texts, use_doc_indices, base_dir):
    
    print("loading bm25 doc indices..." if use_doc_indices else "loading bm25 passage indices...")
    
    if use_doc_indices:
        parent_dir = f"{base_dir}/doc_indices"
        indices_path = f'{parent_dir}/from_raw_texts' if use_raw_texts else f'{parent_dir}/from_clean_texts'
    else:
        parent_dir = f"{base_dir}/passage_indices"
        indices_path = f'{parent_dir}/from_raw_texts' if use_raw_texts else f'{parent_dir}/from_clean_texts'
    
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(indices_path)

    searcher.set_bm25(0.001, 0.5)
    print("Running bm25 retrieval...")

    claims = dict()

    for d in tqdm(data):
        query = d["raw_text"] if use_raw_texts else d["clean_text"]
        op = dict()
        op["claim"] = query
        op["label"] = d["label"]
        
        ground_truth_doc_ids = list(d["joint_ids"])
        if use_doc_indices:
            ground_truth_doc_ids = list(set([value.split('_')[0] for value in ground_truth_doc_ids]))

        hits = searcher.search(query, TOPK)
        
        predicted_doc_ids = []

        for i in range(len(hits)):
            doc = searcher.doc(str(hits[i].docid))
            # doc_contents = json.loads(doc.raw())
            predicted_doc_ids.append(doc.id())
        
        op['ground_truth_docs'] = ground_truth_doc_ids
        op['predicted_docs'] = predicted_doc_ids
        
        claims[d["claim_id"]] = op 

        del op
    
    return claims


def get_doc_by_id(doc_ids, use_doc_indices, use_raw_texts):

    dirs = os.path.abspath(os.curdir).split("/")
    curr_path = "/".join(dirs[:-1])
    wiki_df_folder_path = curr_path + ('/preprocessing/wiki_docs_parquets' if use_doc_indices else '/preprocessing/wiki_passages_parquets_2')
    files = [wiki_df_folder_path + "/" + name for name in sorted(os.listdir(wiki_df_folder_path))]

    docs_list = []

    for file in files:
        df = pd.read_parquet(file)
        # Filter the DataFrame to include only rows with joint_id in joint_ids
        filtered_df = df[df['doc_id'].astype(str).isin(doc_ids)] if use_doc_indices else df[df['joint_id'].isin(doc_ids)]

        for index, row in filtered_df.iterrows(): 
            doc_dict = dict() 
            if use_doc_indices:
                doc_dict['wiki_title'] = " ".join(str(row['title']).split("_")) # to replace underscores in title with spaces
                doc_dict['evidence_doc'] = str(row['raw_text']) if use_raw_texts else str(row['clean_text'])
                doc_dict['doc_id'] = str(row['doc_id'])
            else:
                doc_dict['wiki_title'] = " ".join(str(row['title']).split("_")) # to replace underscores in title with spaces
                doc_dict['evidence_sentence'] = str(row['raw_passage_content']) if use_raw_texts else str(row['clean_passage_content'])
                doc_dict['doc_id'] = row['joint_id']
            
            docs_list.append(doc_dict)

    return docs_list


def map_passages(claims, use_doc_indices, use_raw_texts):
    
    # Gather all unique doc_ids
    all_doc_ids = set()
    for value in claims.values():
        all_doc_ids.update(value['ground_truth_docs'])
        all_doc_ids.update(value['predicted_docs'])

    # Perform the lookup once
    doc_lookup = get_doc_by_id(all_doc_ids, use_doc_indices, use_raw_texts)

    # Replace ground_truth_doc_ids and predicted_doc_ids in my_dict with lookup dictionaries
    for key, value in claims.items():
        value['ground_truth_docs'] = [doc for doc in doc_lookup if doc['doc_id'] in value['ground_truth_docs']]
        value['predicted_docs'] = [doc for doc in doc_lookup if doc['doc_id'] in value['predicted_docs']]

    return claims


if __name__ == "__main__":
    
    use_doc_indices = False
    use_raw_texts = False
    on_complete_fever = False
    is_bm25 = False

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
    if is_bm25:
        top_k_docs = bm25_retrieval(data, use_raw_texts, use_doc_indices, base_dir)
    else:
        top_k_docs = dpr_retrieval(data, use_raw_texts, use_doc_indices, base_dir)

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

    print("Replacing docs with document dictionaries")
    final_claims_dict = map_passages(top_k_docs, use_doc_indices, use_raw_texts)

    # STORE RETRIEVAL RESULTS
    with open(output_file_path, 'w') as json_file:
        json.dump(final_claims_dict, json_file)
    
    print("FINISH")
    
    

