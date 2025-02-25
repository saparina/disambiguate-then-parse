import os
import json

from tqdm import tqdm

from dataset.db_utils import check_and_modify_names, create_db, db_to_text, dump_database, dump_database_raw, dump_database_split, dump_database_agg, merge_all_insert_statements_ambiqt

def read_ambiqt(root_directory="/data/ambig-models/src/", output_dir="database_syn", split="validation", make_different_values=False, syn_types=["column", "table"]):
    dataset = []

    syn_database_path = os.path.join(root_directory, f"AmbiQT/db-content/{output_dir}")
    if not os.path.exists(syn_database_path):
        os.mkdir(syn_database_path)

    for syn_type in syn_types:
        if syn_type == "table":
            dir_name = "tbl-synonyms"
        elif syn_type == "column":
            dir_name = "col-synonyms"
        elif syn_type == "split":
            dir_name = "tbl-split"
        elif syn_type == "agg":
            dir_name = "tbl-agg"
        else:
            raise ValueError(f"Invalid synonym type: {syn_type}")

        file_path = os.path.join(root_directory, "AmbiQT/benchmark", dir_name, f"{split}.json")
                     
        cur_syn_val = json.load(open(file_path))

        if syn_type == "table" or syn_type == "column":
            db_dumps = {}
            nl_synonyms = []
            for ex in tqdm(cur_syn_val):
                db_id = ex['db_id']
                synonyms = ex['extra_table_map'] if syn_type == "table" else ex['extra_map']

                syn_name = list(synonyms.keys())
                assert len(syn_name) == 1, synonyms
                syn_name = syn_name[0]

                if syn_type != "table":
                    col_name = list(synonyms[syn_name].keys())
                    assert len(col_name) == 1, synonyms
                    col_name = col_name[0]

                    nl_synonyms = [db_to_text(val) for val in synonyms[syn_name][col_name]]
                    synonyms[syn_name][col_name] = [check_and_modify_names(val) for val in synonyms[syn_name][col_name]]
                    syn_name += f"_{col_name}"
                else:
                    nl_synonyms = [db_to_text(val) for val in synonyms[syn_name]]
                    synonyms[syn_name] = [check_and_modify_names(val) for val in synonyms[syn_name]]


                db_path = os.path.join(root_directory, "AmbiQT/db-content/database", db_id, db_id + '.sqlite')
                new_db_id = f"{db_id}_{syn_type}_{syn_name}"
                syn_db_path = os.path.join(syn_database_path, new_db_id)
                if not os.path.exists(syn_db_path):
                    os.mkdir(syn_db_path)

                syn_db_path = os.path.join(syn_db_path, new_db_id + ".sqlite")
                if os.path.exists(syn_db_path):
                    if new_db_id not in db_dumps:
                        syn_sql_dump = dump_database_raw(syn_db_path)
                        db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)
                else:
                    if new_db_id not in db_dumps:
                        syn_sql_dump = dump_database(db_path, synonyms, syn_type, make_different_values=make_different_values)
                        db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)

                    if not os.path.exists(syn_db_path):
                        create_db(syn_db_path, db_dumps[new_db_id])

                syn_example = {
                                "db_file": syn_db_path,
                                "db_dump": db_dumps[new_db_id],
                                "syn_type": syn_type,
                                "question": ex["question"],
                                "gold_queries": [ex["query1"], ex["query2"]],
                                "is_ambiguous": True,
                                "nl_synonyms": nl_synonyms
                            }
                dataset.append(syn_example)
        elif syn_type == "split":
            db_dumps = {}
            for ex in tqdm(cur_syn_val):
                db_id = ex['db_id']
                split_map = ex['split_map']
                assert len(split_map) == 1, split_map
                tbl1 = list(ex['split_map'].keys())[0]
                tbl2 = list(ex['split_map'].values())[0]
                split_map_str = f"{tbl1}_{tbl2}"

                db_path = os.path.join(root_directory, "AmbiQT/db-content/database", db_id, db_id + '.sqlite')
                new_db_id = f"{db_id}_{syn_type}_{split_map_str}"
                syn_db_path = os.path.join(syn_database_path, new_db_id)
                if not os.path.exists(syn_db_path):
                    os.mkdir(syn_db_path)

                syn_db_path = os.path.join(syn_db_path, new_db_id + ".sqlite")
                if os.path.exists(syn_db_path):
                    if new_db_id not in db_dumps:
                        syn_sql_dump = dump_database_raw(syn_db_path)
                        db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)
                else:
                    syn_sql_dump = dump_database_split(db_path, split_map, ex["schema_without_content"])
                    db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)

                    if not os.path.exists(syn_db_path):
                        create_db(syn_db_path, db_dumps[new_db_id])

                syn_example = {
                                "db_file": syn_db_path,
                                "db_dump": db_dumps[new_db_id],
                                "syn_type": syn_type,
                                "question": ex["question"],
                                "gold_queries": [ex["query1"], ex["query2"]],
                                "is_ambiguous": True,
                                "split_map": split_map
                            }
                dataset.append(syn_example)

        elif syn_type == "agg":
            db_dumps = {}
            for ex in tqdm(cur_syn_val):
                db_id = ex['db_id']
                tbl1 = ex["new_table_name"]
                tbl2 = "_".join(ex['all_cols'])
                split_map_str = f"{tbl1}_{tbl2}"

                db_path = os.path.join(root_directory, "AmbiQT/db-content/database", db_id, db_id + '.sqlite')
                new_db_id = f"{db_id}_{syn_type}_{split_map_str}"
                syn_db_path = os.path.join(syn_database_path, new_db_id)
                if not os.path.exists(syn_db_path):
                    os.mkdir(syn_db_path)

                syn_db_path = os.path.join(syn_db_path, new_db_id + ".sqlite")
                if os.path.exists(syn_db_path):
                    if new_db_id not in db_dumps:
                        syn_sql_dump = dump_database_raw(syn_db_path)
                        db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)
                else:
                    syn_sql_dump = dump_database_agg(db_path, ex["schema_without_content"], ex["all_raw_cols"], ex["all_cols"], ex["new_table_name"])
                    
                    db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)

                    if not os.path.exists(syn_db_path):
                        create_db(syn_db_path, db_dumps[new_db_id])

                syn_example = {
                                "db_file": syn_db_path,
                                "db_dump": db_dumps[new_db_id],
                                "syn_type": syn_type,
                                "question": ex["question"],
                                "gold_queries": [ex["query1"], ex["query2"]],
                                "is_ambiguous": True
                            }
                dataset.append(syn_example)
    return dataset

    

def fix_dataset_dbs(dataset, root_directory="/data/ambig-models/src/", output_dir="database_syn_eval"):

    syn_database_path = os.path.join(root_directory, f"AmbiQT/db-content/{output_dir}")
    if not os.path.exists(syn_database_path):
        os.mkdir(syn_database_path)

    all_synonyms_col = json.load(open(os.path.join(root_directory, f"AmbiQT/benchmark/col-synonyms/two_col_synonyms.json"), "r"))
    all_synonyms_tbl = json.load(open(os.path.join(root_directory, f"AmbiQT/benchmark/tbl-synonyms/two_tbl_synonyms.json"), "r"))

    db_dumps = {}

    new_dataset = []
    for example in tqdm(dataset):
        try:
            db_id_splitted = example["db_file"].replace(".sqlite", "").split("/")[-1]

            if "_column_" in db_id_splitted:
                db_id_splitted = db_id_splitted.split("_column_")
                db_id = db_id_splitted[0]
                syn_type = "column"

                db_id_splitted = db_id_splitted[1].split("_")
                syn_name = db_id_splitted[0]
                col_name = '_'.join(db_id_splitted[1:])
                all_synonyms = all_synonyms_col
            elif "_table_" in db_id_splitted:
                db_id_splitted = db_id_splitted.split("_table_")
                db_id = db_id_splitted[0]
                syn_type = "table"

                syn_name = db_id_splitted[1]
                all_synonyms = all_synonyms_tbl
            elif "_split_" in db_id_splitted:
                example["db_dump_original"] = example["db_dump"]
                new_dataset.append(example)
                continue
            elif "_agg_" in db_id_splitted:
                example["db_dump_original"] = example["db_dump"]
                new_dataset.append(example)
                continue
            else:
                import pdb; pdb.set_trace()
                raise RuntimeError
            
            all_synonyms_db = {}
            for key, val in all_synonyms[db_id].items():
                if isinstance(val, dict):
                    all_synonyms_db[key.lower()] = {}
                    for key2, val2 in val.items():
                        all_synonyms_db[key.lower()][key2.lower()] = [x.lower() for x in val2]
                else:
                    all_synonyms_db[key.lower()] = [x.lower() for x in val]

            if syn_type == "column":
                if syn_name not in all_synonyms_db:
                    for tbl_name in all_synonyms_db.keys():
                        if tbl_name.startswith(syn_name + "_"):
                            tbl_name_short = tbl_name[len(syn_name + "_"):]
                            if col_name.startswith(tbl_name_short):
                                syn_name_full = f"{syn_name}_{col_name}"
                                syn_name = tbl_name
                                col_name = syn_name_full[len(syn_name + "_"):]
                                break

                all_synonyms_db[syn_name][col_name] = [check_and_modify_names(val) for val in all_synonyms_db[syn_name][col_name]]
            else:
                all_synonyms_db[syn_name] = [check_and_modify_names(val) for val in all_synonyms_db[syn_name]]

            
            if syn_type == "column":
                synonyms = {syn_name: {col_name: all_synonyms_db[syn_name][col_name]}}
            else:
                synonyms = {syn_name: all_synonyms_db[syn_name]}

            db_path = os.path.join(root_directory, "AmbiQT/db-content/database", db_id, db_id + '.sqlite')
            new_db_id = f"{db_id}_{syn_type}_{syn_name}"
            new_db_id += f"_{col_name}" if syn_type == "column" else ""

            syn_db_path = os.path.join(syn_database_path, new_db_id)
            if not os.path.exists(syn_db_path):
                os.mkdir(syn_db_path)
            syn_db_path = os.path.join(syn_db_path, new_db_id + ".sqlite")

            if os.path.exists(syn_db_path):
                if new_db_id not in db_dumps:
                    syn_sql_dump = dump_database_raw(syn_db_path)
                    db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)
            else:
                if new_db_id not in db_dumps:
                    syn_sql_dump = dump_database(db_path, synonyms, syn_type, make_different_values=True)
                    db_dumps[new_db_id] = merge_all_insert_statements_ambiqt(syn_sql_dump)

                if not os.path.exists(syn_db_path):
                    create_db(syn_db_path, db_dumps[new_db_id])

            example["db_file"] = syn_db_path
            example["db_dump_original"] = example["db_dump"]
            example["db_dump"] = db_dumps[new_db_id]
            new_dataset.append(example)
        except:
            pass

    return new_dataset