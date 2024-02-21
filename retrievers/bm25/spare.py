import json
input_data_path = "retrieved_docs/top_10.json"

with open(input_data_path) as f:
        data = json.load(f)

def extract_actual_predicted_ids(data):
    actual = []
    predicted = []

    # Iterate over claims
    for claim_data in data["claims"]:
        # Extract actual_doc_id
        actual_doc_id = claim_data["actual_doc_id"]
        if actual_doc_id == "":
            continue
        actual.append(actual_doc_id)

        # Extract doc_ids while preserving order based on rank
        doc_ids = [doc["doc_id"] for doc in sorted(claim_data["docs"], key=lambda x: x["rank"])]
        predicted.append(doc_ids)
    
    return actual, predicted

actual, predicted = extract_actual_predicted_ids(data)
# Print the results
print("Actual doc_ids:", len(actual))
print("Predicted doc_ids:", len(predicted))

def precision_at_k(actual, predicted, k=10):
     scores = []
     for i in range(len(actual)):
         top_k_pred = predicted[i][:k]
         act_set = {actual[i]}
         pred_set = set(top_k_pred)
         result = len(act_set & pred_set) / len(pred_set)
         scores.append(result)
     return sum(scores) / len(scores)


print(precision_at_k(actual, predicted))