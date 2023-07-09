import json
from tqdm import tqdm
from rdflib.namespace import FOAF
from rdflib import Graph, Literal, RDF, URIRef

# create graph object to store datascience RDF triples.
ds_kg = Graph()
# load semantic types and create mapping from type label to URIRef
node_types_map = {}
for k, v in json.load(open("./data/DS_KB/semantic_types.json")).items():
    node_types_map[k] = URIRef(f"http://example.org/node_type/{v}")
# # load mapping of relation types and create mapping from type label to URIRef.
# edge_types_map = {}
# for k, v in json.load(open("./data/DS_KB/relation_types.json")).items():
#     edge_types_map[k] = URIRef(f"http://example.org/edge_type/{v}")

all_graphs = []
unified_ds_textbook_KG = json.load(open("./data/DS_TextBooks/unified_triples.json"))
all_graphs.append(unified_ds_textbook_KG)
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
with open("./data/DS_KB/all_nodes.json", "w") as f:
    json.dump(all_nodes, f, indent=4, ensure_ascii=False)

# for s, p, o in g:
#     print((s, p, o))

# main
if __name__ == "__main__":
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
    'find approaches that can model "234" which is "Learning to Drive an Autonomous Vehicle".': """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name
        WHERE {
            ?p et:can_model n:234 .

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
    "anything related to decision trees": """
        PREFIX n: <http://example.org/>
        PREFIX nt: <http://example.org/node_type/>
        PREFIX et: <http://example.org/edge_type/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name ?q
        WHERE {
            ?p ?q n:237 .

            ?p foaf:name ?name .
        }
    """}
    rev_nodes = {v: k for k,v in all_nodes.items()}
    
    # queries and intents
    intents = list(sparql_queries.keys())
    queries = list(sparql_queries.values())
    
    # Apply the query to the graph and iterate through results
    # for r in ds_kg.query(sparql_queries[0]): print(r["name"])
    print(f"\x1b[34;1m{intents[1]}\x1b[0m")
    for r in ds_kg.query(queries[1]): print(r["name"])
    print()
    print(f"\x1b[34;1m{intents[2]}\x1b[0m") 
    for r in ds_kg.query(queries[2]):
        q = rev_nodes[int(r["q"].split("/")[-1])]
        print(f'{r["name"]} models {q}')
    print()
    print(f"\x1b[34;1m{intents[3]}\x1b[0m") 
    for r in ds_kg.query(queries[3]):
        q = r["q"].split("/")[-1].replace("_"," ").upper()
        print(f'{r["name"]} {q} Decision Tree')