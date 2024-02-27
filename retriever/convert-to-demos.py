import json
import os
import pandas as pd

def lookup_docs(joint_ids, wiki_df_folder_path):
    
    files = [wiki_df_folder_path+"/"+name for name in sorted(os.listdir(wiki_df_folder_path))]
    predicted_docs_list = []
    
    for file in files:
        df = pd.read_parquet(file)
        # Filter the DataFrame to include only rows with joint_id in joint_ids
        filtered_df = df[df['joint_id'].isin(joint_ids)]

        for index, row in filtered_df.iterrows(): 
            doc_dict = dict()
            doc_dict['title'] = " ".join(str(row['title']).split("_")) # to replace underscores in title with spaces 
            doc_dict['text'] = str(row['raw_passage_content'])
            predicted_docs_list.append(doc_dict)

    return predicted_docs_list


def convert_to_demos(data, use_doc_indices, wiki_df_folder_path, is_bm25):

    claims_list = data['claims']
    x = claims_list[:10]
    op = dict()
    doc_dicts = []

    for claim in claims_list:
        new_dict = dict()
        new_dict["question"] = claim["claim"]
        actual_ids = claim["actual_ids"]
        # print(claim)
        '''
        new_dict["answer"] = SUPPORTS/REFUTES from Ground truth followed by the first occuring ground truth. else NOE.
        '''
        joint_ids = [doc['joint_id'] for doc in claim['docs']]
        new_dict["demos"] = lookup_docs(joint_ids, wiki_df_folder_path)
        print(new_dict)
        doc_dicts.append(new_dict)
        break

    return doc_dicts



if __name__ == "__main__":
    
    use_raw_texts = True
    use_doc_indices = False
    is_bm25 = False
    convert_to_demos = True
    
    base_dir = 'bm25' if is_bm25 else 'dpr'

    if use_doc_indices:
        print("Converting from raw docs" if use_raw_texts else "Converting from clean docs")
    else:
        print("Converting from raw passages" if use_raw_texts else "Converting from clean passages")

    if use_doc_indices:
        parent_dir = f"{base_dir}/retrieved_docs"
        read_path = f"{parent_dir}/top_10_raw_texts_1000.json" if use_raw_texts else f"{parent_dir}/top_10_clean_texts_1000.json"
        save_path = f"{parent_dir}/{'demos' if convert_to_demos else 'new'}_raw_texts_1000.json" if use_raw_texts else f"{parent_dir}/{'demos' if convert_to_demos else 'new'}_clean_texts_1000.json"
    else:
        parent_dir = f"{base_dir}/retrieved_passages"
        read_path = f"{parent_dir}/top_10_raw_texts_1000.json" if use_raw_texts else f"{parent_dir}/top_10_clean_texts_1000.json"
        save_path = f"{parent_dir}/{'demos' if convert_to_demos else 'new'}_raw_texts_1000.json" if use_raw_texts else f"{parent_dir}/{'demos' if convert_to_demos else 'new'}_clean_texts_1000.json"

    with open(read_path) as f:
        data = json.load(f)

    dirs = os.path.abspath(os.curdir).split("/")
    curr_path = "/".join(dirs[:-1])
    wiki_df_folder_path = curr_path + ('/preprocessing/wiki_docs_parquets' if use_doc_indices else '/preprocessing/wiki_passages_parquets_2')
    
    converted_data = convert_to_demos(data, use_doc_indices, wiki_df_folder_path, is_bm25)

    # print(converted_data)

    with open(save_path, 'w') as json_file:
        json.dump(converted_data, json_file)

