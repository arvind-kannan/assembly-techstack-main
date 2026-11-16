import xml.etree.ElementTree as ET
from pathlib import Path

from diss_assembly_dataclasses.assembly import Assembly, o3d_save_triangle_mesh
from diss_assembly_dataclasses.connection import Screw_Connection
from diss_assembly_dataclasses.part import Part

xml_path = Path(__file__).resolve().parent / "BG3" / "BG3.xml"

tree = ET.parse(xml_path)
root = tree.getroot()

asm = Assembly(root.attrib["name"])

# 1) load parts only
for part_element in root.findall("part"):
    asm.add_part(Part.from_xml_element(part_element, xml_path.parent))

# 2) regenerate geometric connections from meshes
asm.find_contacts()

# 3) re-add screw connections from xml
for conn_element in root.findall("connection"):
    if conn_element.attrib["type"] == "screw":
        asm.add_connection(Screw_Connection.from_xml_element(conn_element))

# 4) save meshes manually with Open3D
out_xml = xml_path.with_name("BG3_new.xml")
stl_dir = out_xml.parent / "stl"
stl_dir.mkdir(parents=True, exist_ok=True)

for part in asm.parts.values():
    filename = stl_dir / f"{part.name}.stl"

    if not part.mesh.has_triangle_normals():
        part.mesh.compute_triangle_normals()
    if not part.mesh.has_vertex_normals():
        part.mesh.compute_vertex_normals()

    o3d_save_triangle_mesh(part.mesh, str(filename))
    part.mesh_path = str(filename)

# 5) save xml + npy
asm.save_as_xml(str(out_xml))
