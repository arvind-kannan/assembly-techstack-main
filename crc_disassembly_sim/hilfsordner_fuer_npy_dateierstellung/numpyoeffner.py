import numpy as np
import os
import shutil

def copy_and_convert_npy_to_csv(npy_file, output_dir):
    """
    Kopiert eine .npy-Datei und speichert ihren Inhalt als .csv-Datei.
    :param npy_file: Pfad zur .npy-Datei
    :param output_dir: Verzeichnis, in das die Kopie und CSV gespeichert werden
    """
    if not os.path.exists(npy_file):
        print(f"Fehler: Datei {npy_file} nicht gefunden.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Kopiere die .npy-Datei
    npy_copy = os.path.join(output_dir, os.path.basename(npy_file))
    shutil.copy(npy_file, npy_copy)
    print(f"Kopie gespeichert als: {npy_copy}")
    
    # Lade die Datei
    data = np.load(npy_file, allow_pickle=True)
    
    # Speichere als CSV
    csv_file = os.path.join(output_dir, os.path.splitext(os.path.basename(npy_file))[0] + ".csv")
    np.savetxt(csv_file, data, delimiter=",", fmt="%s")  # fmt="%s" für gemischte Datentypen
    print(f"CSV gespeichert als: {csv_file}")

# Beispiel: Dateipfade anpassen
npy_file = r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sim\spawn_point_faga.npy"  # Pfad zur Eingabedatei
output_dir = r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sim\output"  # Ordner für die Ausgabe
copy_and_convert_npy_to_csv(npy_file, output_dir)
