import pickle
import sys

# Optional: Pfad zu operations.py hinzufügen, falls nicht im Python-Pfad
# import sys
# sys.path.append('Pfad/zum/Ordner/mit/operations.py')

def main(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)


    #if pkl_path.endswith("_mdp.pkl"):
    print("MDP-Graph.")
    print("Typ:", type(obj))
    print("Inhalt:")
    print(obj)
    print(f" obj.nodes: {getattr(obj, 'nodes', None)}")
    print(f" obj.edges: {getattr(obj, 'edges', None)}")


    """if hasattr(obj, 'nodes'):
        for node in obj.nodes(data=True):
            print(node[0], type(node[0]))"""

    if hasattr(obj, 'edges'):
        for edge in obj.edges(data=True):
            
            print(edge[2]['operation'])
            if type(edge[2]['operation']) is not str:
                print(f"Path duration --> {edge[2]['operation'].data['path_duration']} ")

    """else:
        print("No MDP-Graph.")
        print("Typ:", type(obj))
        print("Inhalt:")
        print(obj)
        print(f" obj.nodes: {getattr(obj, 'nodes', None)}")
        print(f" obj.edges: {getattr(obj, 'edges', None)}")
        if hasattr(obj, 'edges'):
            for edge in obj.edges(data=True):
                #print(edge, " : " , edge[2]['operation'].data)
                print(edge[2]['operation'])
                print(f" Path duration --> {edge[2]['operation'].data['path_duration']} ")
        else:
            print("Das Objekt hat keine 'edges'-Methode oder -Attribut.")"""
    
if __name__ == "__main__":
    main(r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sequence_generator\Verifizierungsmodelle\BG1_2\fullgraph_nomilling\fullgraph_nomilling.pkl")
    #main(r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sequence_generator\Verifizierungsmodelle\BG1_2\fullgraph_nomilling\fullgraph_nomilling_mdp.pkl")