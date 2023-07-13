import sys
import json
from tqdm import tqdm
from rdflib.namespace import FOAF
from rdflib import Graph, Literal, RDF, URIRef

def convert_to_triples(wiki_graph):
    triples = []
    for sub,v in wiki_graph.items():
        for obj,e in v["E"]:
            triples.append({
                "sub": (sub,"U",""),
                "obj": (obj,"U",""),
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
                    all_nodes[subn] = (global_ctr, sub[1], {graph_source: None})
                    sub_id = global_ctr
                    sub_node = URIRef(f"http://example.org/{sub_id}")
                    sub_node_type = node_types_map[sub[1]]
                    ds_kg.add((sub_node, RDF.type, URIRef(sub_node_type)))
                    ds_kg.add((sub_node, FOAF.name, Literal(subn)))
                    global_ctr += 1
                else: # subject node already exists.
                    sub_id = all_nodes[subn][0]
                    all_nodes[subn][-1][graph_source] = None
                    sub_node = URIRef(f"http://example.org/{sub_id}")
                if objn not in all_nodes:
                    all_nodes[objn] = (global_ctr, sub[1], {graph_source: None})
                    obj_id = global_ctr
                    obj_node = URIRef(f"http://example.org/{obj_id}")
                    obj_node_type = node_types_map[obj[1]]
                    # print(obj_node_type)
                    ds_kg.add((obj_node, RDF.type, URIRef(obj_node_type)))
                    ds_kg.add((obj_node, FOAF.name, Literal(objn)))
                    global_ctr += 1
                else: # object node already exists.
                    obj_id = all_nodes[objn][0]
                    all_nodes[objn][-1][graph_source] = None
                    obj_node = URIRef(f"http://example.org/{obj_id}")
                edge_type = e.lower().strip().replace("(","").replace(")","").replace(" ","_")
                rel = URIRef(f"http://example.org/edge_type/{edge_type}")
                ds_kg.add((sub_node, rel, obj_node))
                
                # print(sub_node, rel, obj_node)
        with open(weights_save_path, "w") as f:
            json.dump(edge_weights, f, indent=4, ensure_ascii=False)
        with open(nodes_save_path, "w") as f:
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
    'find models that address Sentiment Analysis as a task along with the metrics': """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?model_name ?metric_name ?dataset_name
        WHERE {
            ?model et:can_model n:"""+str(all_nodes['sentiment analysis'][0])+""" .
            ?dataset et:has_goals n:"""+str(all_nodes['sentiment analysis'][0])+""" .
            ?dataset et:evaluated_by ?metric .
            ?model et:has_score ?metric .
            ?dataset foaf:name ?dataset_name .
            ?metric foaf:name ?metric_name .
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
    """}

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

    # eg_results[intents[10]] = [] 
    # print(f"\x1b[34;1m{intents[10]}\x1b[0m") 
    # for r in ds_kg.query(queries[10]):
    #     M = r["model_name"]
    #     E = r["metric_name"]
    #     D = r["dataset_name"]
    #     weight = edge_weights[f"{M}::{E}"]
    #     print(M, D, E, weight)

    eg_results[intents[11]] = [] 
    print(f"\x1b[34;1m{intents[11]}\x1b[0m") 
    for r in ds_kg.query(queries[11]):
        print(r["name"])
        eg_results[intents[11]].append(r["name"])

    eg_results_save_path = "./data/DS_KB/eg_query_results.json"
    with open(eg_results_save_path, "w") as f:
        json.dump(eg_results, f, indent=4)