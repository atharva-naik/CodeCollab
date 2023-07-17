import sys
import json
from typing import *
from tqdm import tqdm
from rdflib.namespace import FOAF
from rdflib import Graph, Literal, RDF, URIRef

def convert_to_triples(wiki_graph):
    triples = []
    node_classes = {}
    for k, v in json.load(open("./data/DS_KB/wikidata_pred_node_classes.json")).items():
        node_classes[k.lower()] = v
    for k, v in json.load(open("./data/DS_KB/wikidata_node_unclassified_preds_sbert_knn.json")).items():
        node_classes[k.lower()] = v
    for sub,v in wiki_graph.items():
        sub = sub.lower()
        sub_sem_type = node_classes[sub]
        for obj,e in v["E"]:
            obj = obj.lower()
            obj_sem_type = node_classes[obj]
            triples.append({
                "sub": (sub,sub_sem_type,""),
                "obj": (obj,obj_sem_type,""),
                "e": e
            })

    return triples

# load semantic types and create mapping from type label to URIRef
node_types_map = {}
for k, v in json.load(open("./data/DS_KB/semantic_types.json")).items():
    node_types_map[k] = URIRef(f"http://example.org/node_type/{v}")
# # load mapping of relation types and create mapping from type label to URIRef.
# edge_types_map = {}
# for k, v in json.load(open("./data/DS_KB/relation_types.json")).items():
#     edge_types_map[k] = URIRef(f"http://example.org/edge_type/{v}")

def load_ds_kg(do_reset: bool, save_path: str="./data/DS_KB/rdf_triples_turtle.txt",
               weights_save_path: str="./data/DS_KB/all_edge_weights.json",
               nodes_save_path: str="./data/DS_KB/all_nodes.json"):
    # create graph object to store datascience RDF triples.
    ds_kg = Graph()
    edge_weights = {}
    if do_reset:
        all_graphs = []
        unified_ds_textbook_KG = json.load(open("./data/DS_TextBooks/unified_triples.json"))
        all_graphs.append((unified_ds_textbook_KG, "DS TextBooks"))
        pwc_papers_KG = json.load(open("./data/PwC/unified_pwc_triples.json"))
        all_graphs.append((pwc_papers_KG, "PwC Papers"))
        wikidata_KG = convert_to_triples(json.load(open("./data/WikiData/ds_qpq_graph_pruned.json")))
        all_graphs.append((wikidata_KG, "WikiData"))
        global_ctr = 1
        all_nodes = {}
        for graph, graph_source in all_graphs:
            for triple in tqdm(graph):
                # Add triples using store's add() method.
                sub = triple["sub"]
                obj = triple["obj"]
                subn = sub[0].lower()
                objn = obj[0].lower()
                e = triple["e"]
                weight = triple.get("w")
                if weight is not None:
                    edge_weights[subn+"::"+objn] = weight
                if subn not in all_nodes:
                    all_nodes[subn] = (global_ctr, sub[1], set(), set())
                    if sub[2].strip() != "": 
                        all_nodes[subn][-2].add(sub[2])
                    all_nodes[subn][-1].add(graph_source)
                    sub_id = global_ctr
                    sub_node = URIRef(f"http://example.org/{sub_id}")
                    sub_node_type = node_types_map[sub[1]]
                    ds_kg.add((sub_node, RDF.type, URIRef(sub_node_type)))
                    ds_kg.add((sub_node, FOAF.name, Literal(subn)))
                    global_ctr += 1
                else: # subject node already exists.
                    sub_id = all_nodes[subn][0]
                    if sub[2].strip() != "": 
                        all_nodes[subn][-2].add(sub[2])
                    all_nodes[subn][-1].add(graph_source)
                    sub_node = URIRef(f"http://example.org/{sub_id}")
                if objn not in all_nodes:
                    all_nodes[objn] = (global_ctr, obj[1], set(), set())
                    if obj[2].strip() != "":
                        all_nodes[objn][-2].add(obj[2]) 
                    all_nodes[objn][-1].add(graph_source)
                    obj_id = global_ctr
                    obj_node = URIRef(f"http://example.org/{obj_id}")
                    obj_node_type = node_types_map[obj[1]]
                    # print(obj_node_type)
                    ds_kg.add((obj_node, RDF.type, URIRef(obj_node_type)))
                    ds_kg.add((obj_node, FOAF.name, Literal(objn)))
                    global_ctr += 1
                else: # object node already exists.
                    obj_id = all_nodes[objn][0]
                    if obj[2].strip() != "": 
                        all_nodes[objn][-2].add(obj[2])
                    all_nodes[objn][-1].add(graph_source)
                    obj_node = URIRef(f"http://example.org/{obj_id}")
                edge_type = e.lower().strip().replace("(","").replace(")","").replace(" ","_")
                rel = URIRef(f"http://example.org/edge_type/{edge_type}")
                ds_kg.add((sub_node, rel, obj_node))
                
                # print(sub_node, rel, obj_node)
        with open(weights_save_path, "w") as f:
            json.dump(edge_weights, f, indent=4, ensure_ascii=False)
        with open(nodes_save_path, "w") as f:
            all_nodes_serialized = {}
            for k,v in all_nodes.items():
                all_nodes_serialized[k] = [
                    v[0], v[1],
                    list(v[2]),
                    list(v[3])
                ] 
            all_nodes = all_nodes_serialized
            json.dump(all_nodes, f, indent=4, ensure_ascii=False)
        with open(save_path, "w") as f:
            f.write(ds_kg.serialize(format='turtle'))
    else: 
        all_nodes = json.load(open(nodes_save_path))
        edge_weights = json.load(open(weights_save_path))
        ds_kg.parse(save_path)
    
    return ds_kg, all_nodes, edge_weights

# for s, p, o in g:
#     print((s, p, o))

# main
if __name__ == "__main__":
    do_reset = False
    if len(sys.argv) > 1 and sys.argv[1] == "reset": do_reset = True
    ds_kg, all_nodes, edge_weights = load_ds_kg(do_reset=do_reset)
    print(f"|V|: {len(all_nodes)}, |E|: {len(ds_kg)}, |w|: {len(edge_weights)}")
    # print(ds_kg.serialize(format='turtle'))
    sparql_queries = {
    "find all model names.": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name
        WHERE {
            ?p rdf:type nt:model .

            ?p foaf:name ?name .
        }
    """,
    'find approaches related to "Autonomous Driving".': """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?collection
        WHERE {
            ?p et:can_model n:"""+str(all_nodes['autonomous driving'][0])+""" .
            ?p et:subclass_of ?q .
            ?q foaf:name ?collection .
            ?p foaf:name ?name .
        }
    """,
    "find approaches for whom we have a KB record for modeling a specific task.": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p et:can_model ?q .

            ?p foaf:name ?name .
        }
    """,
    "anything related to Decision Tree Learning": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p ?q n:"""+str(all_nodes['decision tree learning'][0])+""" .

            ?p foaf:name ?name .
        }
    """,
    "anything related to CutMix": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p ?q n:"""+str(all_nodes['cutmix'][0])+""" .

            ?p foaf:name ?name .
        }
    """,
    "anything related to Path Planning": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p ?q n:"""+str(all_nodes['path planning'][0])+""" .

            ?p foaf:name ?name .
        }
    """,
    "anything related to Capsule Network": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p ?q n:"""+str(all_nodes['capsule network'][0])+""" .

            ?p foaf:name ?name .
        }
    """,
    "modeling approaches for sentiment classification": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p et:can_model n:"""+str(all_nodes['sentiment classification'][0])+""" .

            ?p foaf:name ?name .
        }
    """,
    'find loss functions related to "Autonomous Driving".': """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name
        WHERE {
            ?p et:can_model n:"""+str(all_nodes['autonomous driving'][0])+""" .
            ?p et:subclass_of n:"""+str(all_nodes['loss functions'][0])+""" .
            ?p foaf:name ?name .
        }
    """,
    'find CNNs related to "Autonomous Driving".': """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name
        WHERE {
            ?p et:can_model n:"""+str(all_nodes['autonomous driving'][0])+""" .
            ?p et:subclass_of n:"""+str(all_nodes['convolutional neural networks'][0])+""" .
            ?p foaf:name ?name .
        }
    """,
    'find models that address Sentiment Analysis as a task and pick the best model for each dataset by Accuracy': """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?model_name ?dataset_name
        WHERE {
            ?dataset et:has_goals n:"""+str(all_nodes['sentiment analysis'][0])+""" .
            ?model et:has_score_for ?dataset .
            ?dataset foaf:name ?dataset_name .
            ?model foaf:name ?model_name .
        }
    """,
    "techniques that handle missing values": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name
        WHERE {
            ?p et:handles n:"""+str(all_nodes['missing values'][0])+""" .
            ?p foaf:name ?name .
        }
    """,
    "steps involved in defining a neural network":"""
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?top_level_step
        WHERE {
            n:"""+str(all_nodes['define neural network'][0])+""" et:has_steps ?p .
            ?p foaf:name ?top_level_step .
        }
    """,
    "steps invloved in loading PyTorch data": "",
    "find ancestors of a node": "",
    "steps involved in developing PyTorch DataLoaders": ""}

    def recursively_find_ancestors(name: str, kg, rel_types: List[str]=["subclass_of", "instance_of"]):
        query = """ PREFIX n: <http://example.org/>
            PREFIX nt: <http://example.org/node_type/>
            PREFIX et: <http://example.org/edge_type/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name
            WHERE {
                n:"""+str(all_nodes[name][0])+""" et:{} ?p .
                ?p foaf:name ?name .
            }
        """
        ancestors = {}
        for rel_type in rel_types:
            ancestors[rel_type] = {}
            for r in kg.query(query.format(rel_type)):
                name = r["name"].toPython()
                ancestors[rel_type][name] = recursively_find_ancestors(name, kg)

        return ancestors

    def recursively_find_steps(name: str, kg, all_nodes: dict):
        query = """ PREFIX n: <http://example.org/>
            PREFIX nt: <http://example.org/node_type/>
            PREFIX et: <http://example.org/edge_type/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?top_level_step
            WHERE {
                n:"""+str(all_nodes[name][0])+""" et:has_steps ?p .
                ?p foaf:name ?top_level_step .
            }
        """
        steps = {}
        for r in kg.query(query):
            step_name = r["top_level_step"].toPython()
            steps[step_name] = {}
            steps[step_name]["description"] = all_nodes[step_name][-2]
            steps[step_name]["children"] = recursively_find_steps(
                step_name, kg, all_nodes
            )

        return steps

    rev_nodes = {v[0]: k for k,v in all_nodes.items()}
    eg_results = {}

    # queries and intents
    intents = list(sparql_queries.keys())
    queries = list(sparql_queries.values())
    
    # Apply the query to the graph and iterate through results
    # for r in ds_kg.query(sparql_queries[0]): print(r["name"])

    ctr = 0
    eg_results[intents[1]] = []
    # print(f"\x1b[34;1m{intents[1]}\x1b[0m")    
    for r in ds_kg.query(queries[1]): 
        eg_results[intents[1]].append((r["name"], r["collection"]))
        # print(r["name"], r["collection"])
        ctr += 1
    # print(f"got {ctr} hits\n")

    # print(f"\x1b[34;1m{intents[2]}\x1b[0m") 
    # for r in ds_kg.query(queries[2]):
    #     q = rev_nodes[int(r["q"].split("/")[-1])]
    #     print(f'{r["name"]} models {q}')
    # print()
    eg_results[intents[3]] = [] 
    # print(f"\x1b[34;1m{intents[3]}\x1b[0m") 
    for r in ds_kg.query(queries[3]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        # print(f'{r["name"]} {q} Decision Tree Learning')
        eg_results[intents[3]].append(f'{r["name"]} {q} Decision Tree Learning')

    eg_results[intents[4]] = [] 
    # print(f"\x1b[34;1m{intents[4]}\x1b[0m") 
    for r in ds_kg.query(queries[4]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        # print(f'{r["name"]} {q} CutMix')
        eg_results[intents[4]].append(f'{r["name"]} {q} CutMix')

    eg_results[intents[5]] = [] 
    # print(f"\x1b[34;1m{intents[5]}\x1b[0m") 
    for r in ds_kg.query(queries[5]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        # print(f'{r["name"]} {q} Path Planning')
        eg_results[intents[5]].append(f'{r["name"]} {q} Path Planning')

    eg_results[intents[6]] = [] 
    # print(f"\x1b[34;1m{intents[6]}\x1b[0m") 
    for r in ds_kg.query(queries[6]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        # print(f'{r["name"]} {q} Capsule Network')
        eg_results[intents[6]].append(f'{r["name"]} {q} Capsule Network')

    eg_results[intents[7]] = [] 
    # print(f"\x1b[34;1m{intents[7]}\x1b[0m") 
    for r in ds_kg.query(queries[7]):
        # print(f'{r["name"]} CAN MODEL Sentiment Classification')
        eg_results[intents[7]].append(f'{r["name"]} CAN MODEL Sentiment Classification')

    eg_results[intents[8]] = [] 
    # print(f"\x1b[34;1m{intents[8]}\x1b[0m") 
    for r in ds_kg.query(queries[8]):
        # print(r["name"])
        eg_results[intents[8]].append(r["name"])

    eg_results[intents[9]] = [] 
    # print(f"\x1b[34;1m{intents[9]}\x1b[0m") 
    for r in ds_kg.query(queries[9]):
        # print(r["name"])
        eg_results[intents[9]].append(r["name"])

    import numpy as np
    from collections import defaultdict
    dataset_wise_model_scores = defaultdict(lambda: {})
    metrics = defaultdict(lambda: 0)
    print(f"\x1b[34;1m{intents[10]}\x1b[0m") 
    for r in ds_kg.query(queries[10]):
        M = r["model_name"]
        D = r["dataset_name"]
        metrics_row = edge_weights[f"{M}::{D}"]
        dataset_wise_model_scores[D][M] = metrics_row
        for k in metrics_row: metrics[k] += 1
        # print(M, D, metrics_row)
    dataset_wise_model_scores = {
        k: list(v.keys())[
            np.argmax([val.get("Accuracy",0) for val in v.values()])
        ] for k,v in dataset_wise_model_scores.items() if "Accuracy" in list(v.values())[0]
    }
    eg_results[intents[10]] = dataset_wise_model_scores
    print(json.dumps(dataset_wise_model_scores, indent=4))
    # print(metrics)
    eg_results[intents[11]] = [] 
    print(f"\x1b[34;1m{intents[11]}\x1b[0m") 
    for r in ds_kg.query(queries[11]):
        print(r["name"])
        eg_results[intents[11]].append(r["name"])

    # print(f"\x1b[34;1m{intents[12]}\x1b[0m") 
    eg_results[intents[12]] = recursively_find_steps(
        "define neural network", 
        ds_kg, all_nodes
    )
    # print(eg_results[intents[12]])

    # print(f"\x1b[34;1m{intents[13]}\x1b[0m") 
    eg_results[intents[13]] = recursively_find_steps(
        "loading data", 
        ds_kg, all_nodes
    )
    # print(eg_results[intents[13]])

    # print(f"\x1b[34;1m{intents[14]}\x1b[0m") 
    # eg_results[intents[14]] = recursively_find_ancestors("model interpretability with captum", ds_kg)
    # print(eg_results[intents[14]])

    print(f"\x1b[34;1m{intents[15]}\x1b[0m") 
    eg_results[intents[15]] = recursively_find_steps(
        "developing custom pytorch dataloaders", 
        ds_kg, all_nodes
    )
    print(json.dumps(eg_results[intents[15]], indent=4))

    eg_results_save_path = "./data/DS_KB/eg_query_results.json"
    with open(eg_results_save_path, "w") as f:
        json.dump(eg_results, f, indent=4)