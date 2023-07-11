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

def load_ds_kg(do_reset: bool, 
               save_path: str="./data/DS_KB/rdf_triples_turtle.txt",
               nodes_save_path: str="./data/DS_KB/all_nodes.json"):
    # create graph object to store datascience RDF triples.
    ds_kg = Graph()
    if do_reset:
        all_graphs = []
        unified_ds_textbook_KG = json.load(open("./data/DS_TextBooks/unified_triples.json"))
        all_graphs.append(unified_ds_textbook_KG)
        pwc_papers_KG = json.load(open("./data/PwC/unified_pwc_triples.json"))
        all_graphs.append(pwc_papers_KG)
        wikidata_KG = convert_to_triples(json.load(open("./data/WikiData/ds_qpq_graph_pruned.json")))
        all_graphs.append(wikidata_KG)
        global_ctr = 1
        all_nodes = {}
        for graph in all_graphs:
            for triple in tqdm(graph):
                # Add triples using store's add() method.
                sub = triple["sub"]
                obj = triple["obj"]
                e = triple["e"]
                if sub[0] not in all_nodes:
                    all_nodes[sub[0]] = global_ctr
                    sub_id = global_ctr
                    sub_node = URIRef(f"http://example.org/{sub_id}")
                    sub_node_type = node_types_map[sub[1]]
                    ds_kg.add((sub_node, RDF.type, URIRef(sub_node_type)))
                    ds_kg.add((sub_node, FOAF.name, Literal(sub[0])))
                    global_ctr += 1
                else: # subject node already exists.
                    sub_id = all_nodes[sub[0]]
                    sub_node = URIRef(f"http://example.org/{sub_id}")
                if obj[0] not in all_nodes:
                    all_nodes[obj[0]] = global_ctr
                    obj_id = global_ctr
                    obj_node = URIRef(f"http://example.org/{obj_id}")
                    obj_node_type = node_types_map[obj[1]]
                    # print(obj_node_type)
                    ds_kg.add((obj_node, RDF.type, URIRef(obj_node_type)))
                    ds_kg.add((obj_node, FOAF.name, Literal(obj[0])))
                    global_ctr += 1
                else: # object node already exists.
                    obj_id = all_nodes[obj[0]]
                    obj_node = URIRef(f"http://example.org/{obj_id}")
                edge_type = e.lower().strip().replace("(","").replace(")","").replace(" ","_")
                rel = URIRef(f"http://example.org/edge_type/{edge_type}")
                ds_kg.add((sub_node, rel, obj_node))
                # print(sub_node, rel, obj_node)
        with open(nodes_save_path, "w") as f:
            json.dump(all_nodes, f, indent=4, ensure_ascii=False)
        with open(save_path, "w") as f:
            f.write(ds_kg.serialize(format='turtle'))
    else: 
        all_nodes = json.load(open(nodes_save_path))
        ds_kg.parse(save_path)
    
    return ds_kg, all_nodes

# for s, p, o in g:
#     print((s, p, o))

# main
if __name__ == "__main__":
    do_reset = False
    if len(sys.argv) > 1 and sys.argv[1] == "reset": do_reset = True
    ds_kg, all_nodes = load_ds_kg(do_reset=do_reset)
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
            ?p et:can_model n:"""+str(all_nodes['Autonomous Driving'])+""" .
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
            ?p ?q n:"""+str(all_nodes['Decision Tree Learning'])+""" .

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
            ?p ?q n:"""+str(all_nodes['CutMix'])+""" .

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
            ?p ?q n:"""+str(all_nodes['Path Planning'])+""" .

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
            ?p ?q n:"""+str(all_nodes['Capsule Network'])+""" .

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
            ?p et:can_model n:"""+str(all_nodes['Sentiment Classification'])+""" .

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
            ?p et:can_model n:"""+str(all_nodes['Autonomous Driving'])+""" .
            ?p et:subclass_of n:"""+str(all_nodes['Loss Functions'])+""" .
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
            ?p et:can_model n:"""+str(all_nodes['Autonomous Driving'])+""" .
            ?p et:subclass_of n:"""+str(all_nodes['Convolutional Neural Networks'])+""" .
            ?p foaf:name ?name .
        }
    """}
    rev_nodes = {v: k for k,v in all_nodes.items()}
    eg_results = {}

    # queries and intents
    intents = list(sparql_queries.keys())
    queries = list(sparql_queries.values())
    
    # Apply the query to the graph and iterate through results
    # for r in ds_kg.query(sparql_queries[0]): print(r["name"])
    print(f"\x1b[34;1m{intents[1]}\x1b[0m")
    ctr = 0
    eg_results[intents[1]] = []
    for r in ds_kg.query(queries[1]): 
        eg_results[intents[1]].append((r["name"], r["collection"]))
        print(r["name"], r["collection"])
        ctr += 1
    print(f"got {ctr} hits\n")
    # print(f"\x1b[34;1m{intents[2]}\x1b[0m") 
    # for r in ds_kg.query(queries[2]):
    #     q = rev_nodes[int(r["q"].split("/")[-1])]
    #     print(f'{r["name"]} models {q}')
    # print()
    eg_results[intents[3]] = [] 
    print(f"\x1b[34;1m{intents[3]}\x1b[0m") 
    for r in ds_kg.query(queries[3]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        print(f'{r["name"]} {q} Decision Tree Learning')
        eg_results[intents[3]].append(f'{r["name"]} {q} Decision Tree Learning')

    eg_results[intents[4]] = [] 
    print(f"\x1b[34;1m{intents[4]}\x1b[0m") 
    for r in ds_kg.query(queries[4]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        print(f'{r["name"]} {q} CutMix')
        eg_results[intents[4]].append(f'{r["name"]} {q} CutMix')

    eg_results[intents[5]] = [] 
    print(f"\x1b[34;1m{intents[5]}\x1b[0m") 
    for r in ds_kg.query(queries[5]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        print(f'{r["name"]} {q} Path Planning')
        eg_results[intents[5]].append(f'{r["name"]} {q} Path Planning')

    eg_results[intents[6]] = [] 
    print(f"\x1b[34;1m{intents[6]}\x1b[0m") 
    for r in ds_kg.query(queries[6]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        print(f'{r["name"]} {q} Capsule Network')
        eg_results[intents[6]].append(f'{r["name"]} {q} Capsule Network')

    eg_results[intents[7]] = [] 
    print(f"\x1b[34;1m{intents[7]}\x1b[0m") 
    for r in ds_kg.query(queries[7]):
        print(f'{r["name"]} CAN MODEL Sentiment Classification')
        eg_results[intents[7]].append(f'{r["name"]} CAN MODEL Sentiment Classification')

    eg_results[intents[8]] = [] 
    print(f"\x1b[34;1m{intents[8]}\x1b[0m") 
    for r in ds_kg.query(queries[8]):
        print(r["name"])
        eg_results[intents[8]].append(r["name"])

    eg_results[intents[9]] = [] 
    print(f"\x1b[34;1m{intents[9]}\x1b[0m") 
    for r in ds_kg.query(queries[9]):
        print(r["name"])
        eg_results[intents[9]].append(r["name"])

    eg_results_save_path = "./data/DS_KB/eg_query_results.json"
    with open(eg_results_save_path, "w") as f:
        json.dump(eg_results, f, indent=4)