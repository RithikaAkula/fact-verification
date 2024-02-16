import json
import os
from tqdm import tqdm

TOPK = 10


def bm25_retrieval(data):
    
    from pyserini.search.lucene import LuceneSearcher

    print("loading bm25 index...")
    searcher = LuceneSearcher('bm25_indices')
    # TO FINE-TUNE:
    # searcher.set_bm25(0.9, 0.4)

    print("running bm25 retrieval...")

    top_k_docs = dict()
    top_k_docs["claims"] = []

    for d in tqdm(data):
        op = dict()
        query = d["claim"]
        op["claim"] = query
        op["actual_doc_id"] = d["doc_id"]
        hits = searcher.search(query, TOPK)
        
        docs = []
        for i in range(len(hits)):
            # print(hits[i].docid)
            doc = searcher.doc(str(hits[i].docid))
            doc_contents = json.loads(doc.raw())
            ids = doc.id().split("_")
            docs.append({
                "rank": i+1,
                "predicted_passage": doc_contents["contents"],
                "predicted_passage_id": ids[1],
                "doc_id": ids[0],
                "score": f"{hits[i].score:.5f}",
            })
        op["docs"] = docs
        top_k_docs["claims"].append(op)
        del op
    
    return top_k_docs


def precision_at_k(actual, predicted, k=10):
     scores = []
     for i in range(len(actual)):
         top_k_pred = predicted[i][:k]
         act_set = set(actual[i])
         pred_set = set(top_k_pred)
         result = len(act_set & pred_set) / len(pred_set)
         scores.append(result)
     return sum(scores) / len(scores)


if __name__ == "__main__":

    output_file_path = f"retrieved_docs/top_{TOPK}.json"

    dirs = os.path.abspath(os.curdir).split("/")
    preprocessing_path = "/".join(dirs[:-2])
    input_data_path = preprocessing_path + '/preprocessing/processed_fever/fever-1000-new.json'


    with open(input_data_path) as f:
        data = json.load(f)

    top_k_docs = bm25_retrieval(data)

    with open(output_file_path, 'w') as json_file:
        json.dump(top_k_docs, json_file)

    # WRITE EVALUATION SCRIPT