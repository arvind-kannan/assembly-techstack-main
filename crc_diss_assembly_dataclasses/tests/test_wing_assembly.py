"""
Test the Open3D + FCL based Assembly class against reference data from the XML file.

This test loads the wing assembly from XML (created with the old trimesh-based implementation)
and compares it with a newly created assembly using load_parts_from_folder.
"""
import os
import pytest
import numpy as np

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part, load_parts_from_folder
from diss_assembly_dataclasses.connection import Geometric_Connection


# Path to test data
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
WING_ASSEMBLY_DIR = os.path.join(TEST_DIR, "wing-assembly")
STL_DIR = os.path.join(WING_ASSEMBLY_DIR, "stl")
DATA_DIR = os.path.join(WING_ASSEMBLY_DIR, "airplane_wing_data")
XML_PATH = os.path.join(WING_ASSEMBLY_DIR, "wing_assembly.xml")


# Mapping from part names (in XML) to STL filenames (without extension)
PART_NAME_TO_STL = {
    "REAR SPAR": "CMT5730B072-041",
    "FWD SPAR": "CMT5730B073-041",
    "UPPER SKIN": "CMT5730B074-041",
    "FWD ROOT FITTING SPLICE PLATE": "CMT5730B076-101",
    "AFT ROOT FITTING SPLICE PLATE": "CMT5730B077-101",
    "MID RIB 1": "CMT5730B078-101",
    "MID RIB 2": "CMT5730B079-101",
    "MID RIB 3": "CMT5730B080-101",
    "LE RIB 1": "CMT5730B081-101",
    "LE RIB 2": "CMT5730B082-101",
    "LE RIB 5": "CMT5730B085-101",
    "LE RIB 6": "CMT5730B086-101",
    "LE RIB 7": "CMT5730B087-101",
    "TE RIB 1": "CMT5730B088-101",
    "TE RIB 2": "CMT5730B089-101",
    "TE RIB 3": "CMT5730B090-101",
    "CLOSE OUT TIP RIB": "CMT5730B091-101",
    "TIP CAP LE": "CMT5730B093-101",
    "TIP CAP TE": "CMT5730B094-101",
    "WIRING LOOM": "CMT5730B096-101",
    "LIGHT LENS": "CMT5730B098-101",
    "LEADING EDGE": "CMT5730B099-101",
}

STL_TO_PART_NAME = {v: k for k, v in PART_NAME_TO_STL.items()}


def load_reference_connections():
    """
    Load the reference connections from the .npy files in the data directory.
    Returns a dict mapping (part1, part2) -> {"points": np.ndarray, "normals": np.ndarray}
    """
    connections = {}
    
    # List all _points.npy files
    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_points.npy"):
            base = filename.replace("_points.npy", "")
            parts = base.split("_")
            if len(parts) == 2:
                stl1, stl2 = parts
                part1 = STL_TO_PART_NAME.get(stl1, stl1)
                part2 = STL_TO_PART_NAME.get(stl2, stl2)
                
                points_file = os.path.join(DATA_DIR, f"{base}_points.npy")
                normals_file = os.path.join(DATA_DIR, f"{base}_normals.npy")
                distances_file = os.path.join(DATA_DIR, f"{base}_distances.npy")
                
                if os.path.exists(points_file) and os.path.exists(normals_file):
                    points = np.load(points_file, allow_pickle=True)
                    normals = np.load(normals_file, allow_pickle=True)
                    distances = np.load(distances_file, allow_pickle=True) if os.path.exists(distances_file) else None
                    
                    # Store with sorted part names as key for consistent lookup
                    key = tuple(sorted([part1, part2]))
                    connections[key] = {
                        "points": np.asarray(points),
                        "normals": np.asarray(normals),
                        "distances": np.asarray(distances) if distances is not None else None,
                        "part1": part1,
                        "part2": part2,
                    }
    
    return connections


def create_assembly_from_parts():
    """
    Create a new assembly by loading parts from the STL folder.
    """
    parts = load_parts_from_folder(STL_DIR, depth=0)
    
    # Rename parts to match the XML naming convention
    for part in parts:
        if part.name in STL_TO_PART_NAME:
            part.name = STL_TO_PART_NAME[part.name]
    
    assembly = Assembly("test_wing_assembly", parts, contact_tollerance=1.0, dot_product_threshold=-0.99)
    return assembly


@pytest.fixture(scope="module")
def reference_connections():
    """Fixture to load reference connections once per module."""
    return load_reference_connections()


@pytest.fixture(scope="module")
def new_assembly():
    """Fixture to create a new assembly and find contacts.
    
    Note: This can take several minutes for 22 parts (231 pair combinations).
    """
    assembly = create_assembly_from_parts()
    print(f"\nFinding contacts for {len(assembly.parts)} parts...")
    assembly.find_contacts()
    print(f"Found {len(assembly.connections)} connections")
    return assembly


@pytest.fixture(scope="module")
def small_assembly():
    """Fixture for a smaller assembly with just 3 parts for fast testing."""
    # Select parts that should have connections based on reference data
    part_stls = ["CMT5730B072-041", "CMT5730B074-041", "CMT5730B078-101"]
    parts = []
    
    for stl_name in part_stls:
        stl_path = os.path.join(STL_DIR, f"{stl_name}.stl")
        if os.path.exists(stl_path):
            part_name = STL_TO_PART_NAME.get(stl_name, stl_name)
            part = Part(part_name, stl_path)
            parts.append(part)
    
    assembly = Assembly("small_test", parts, contact_tollerance=1.0, dot_product_threshold=-0.99)
    print(f"\nFinding contacts for small assembly ({len(assembly.parts)} parts)...")
    import time
    start = time.time()
    assembly.find_contacts()
    print(f"Found {len(assembly.connections)} connections in {time.time()-start:.2f}s")
    return assembly


@pytest.fixture(scope="module")
def two_part_assembly():
    """Fixture for a minimal 2-part assembly for fastest testing."""
    # Just 2 parts = 1 pair to check
    part_stls = ["CMT5730B072-041", "CMT5730B074-041"]  # REAR SPAR and UPPER SKIN
    parts = []
    
    for stl_name in part_stls:
        stl_path = os.path.join(STL_DIR, f"{stl_name}.stl")
        if os.path.exists(stl_path):
            part_name = STL_TO_PART_NAME.get(stl_name, stl_name)
            part = Part(part_name, stl_path)
            parts.append(part)
    
    assembly = Assembly("two_part_test", parts, contact_tollerance=1.0, dot_product_threshold=-0.99)
    print(f"\nFinding contacts for 2-part assembly...")
    import time
    start = time.time()
    assembly.find_contacts()
    print(f"Found {len(assembly.connections)} connections in {time.time()-start:.2f}s")
    return assembly


class TestWingAssemblyParts:
    """Tests for part loading and basic properties."""
    
    def test_stl_directory_exists(self):
        """Test that the STL directory exists."""
        assert os.path.exists(STL_DIR), f"STL directory not found: {STL_DIR}"
    
    def test_data_directory_exists(self):
        """Test that the data directory exists."""
        assert os.path.exists(DATA_DIR), f"Data directory not found: {DATA_DIR}"
    
    def test_load_parts_from_folder(self):
        """Test that parts can be loaded from the STL folder."""
        parts = load_parts_from_folder(STL_DIR, depth=0)
        assert len(parts) == 22, f"Expected 22 parts, got {len(parts)}"
    
    def test_parts_have_meshes(self):
        """Test that all loaded parts have valid meshes."""
        parts = load_parts_from_folder(STL_DIR, depth=0)
        for part in parts:
            assert part.mesh is not None, f"Part {part.name} has no mesh"
            assert len(part.mesh.vertices) > 0, f"Part {part.name} has no vertices"
            assert len(part.mesh.triangles) > 0, f"Part {part.name} has no triangles"


class TestTwoPartAssembly:
    """Fastest tests using just 2 parts (1 pair to check)."""
    
    def test_two_part_assembly_finds_contact(self, two_part_assembly):
        """Test contact detection between REAR SPAR and UPPER SKIN."""
        assert len(two_part_assembly.connections) > 0, "No connection found between REAR SPAR and UPPER SKIN"
        print(f"\nTwo-part assembly found {len(two_part_assembly.connections)} connections")
    
    def test_two_part_contact_has_points(self, two_part_assembly):
        """Test that the detected contact has valid points."""
        for conn in two_part_assembly.connections:
            if isinstance(conn, Geometric_Connection):
                assert len(conn.points) > 0, "Connection has no contact points"
                assert len(conn.normals) == len(conn.points), "Mismatched points/normals count"
                print(f"\nConnection has {len(conn.points)} contact points")


class TestSmallAssemblyQuick:
    """Quick tests using a small 3-part assembly for fast iteration."""
    
    @pytest.mark.slow
    def test_small_assembly_finds_contacts(self, small_assembly):
        """Test that the small assembly finds contacts between the 3 parts."""
        # REAR SPAR, UPPER SKIN, MID RIB 1 should all have connections
        assert len(small_assembly.connections) > 0, "No connections found in small assembly"
        print(f"\nSmall assembly found {len(small_assembly.connections)} connections")
    
    @pytest.mark.slow
    def test_small_assembly_expected_pairs(self, small_assembly, reference_connections):
        """Test that expected connection pairs are found in small assembly."""
        expected_pairs = [
            ("REAR SPAR", "UPPER SKIN"),
            ("REAR SPAR", "MID RIB 1"),
            ("UPPER SKIN", "MID RIB 1"),
        ]
        
        found_pairs = set()
        for conn in small_assembly.connections:
            if isinstance(conn, Geometric_Connection):
                key = tuple(sorted([conn.part_name1, conn.part_name2]))
                found_pairs.add(key)
        
        print(f"\nFound pairs: {found_pairs}")
        
        for pair in expected_pairs:
            key = tuple(sorted(pair))
            if key in reference_connections:
                # This pair exists in reference, so we should find it
                assert key in found_pairs, f"Expected connection {pair} not found"
    
    @pytest.mark.slow
    def test_small_assembly_contact_points_valid(self, small_assembly):
        """Test that contact points have valid coordinates and normals."""
        for conn in small_assembly.connections:
            if isinstance(conn, Geometric_Connection):
                points = np.asarray(conn.points)
                normals = np.asarray(conn.normals)
                
                assert len(points) > 0, f"Connection {conn} has no points"
                assert len(normals) == len(points), f"Connection {conn} has mismatched points/normals"
                
                # Check normals are roughly unit length
                norms = np.linalg.norm(normals, axis=1)
                assert np.all(norms > 0.5), f"Connection {conn} has zero-length normals"


class TestWingAssemblyConnections:
    """Tests for contact detection and connection comparison.
    
    NOTE: These tests use the full 22-part assembly and are SLOW (several minutes).
    Run with: pytest -m "not slow" to skip these.
    """
    
    def test_reference_connections_loaded(self, reference_connections):
        """Test that reference connections are loaded from .npy files."""
        assert len(reference_connections) > 0, "No reference connections loaded"
        print(f"\nLoaded {len(reference_connections)} reference connections")
    
    @pytest.mark.slow
    def test_assembly_finds_contacts(self, new_assembly):
        """Test that the new assembly finds some contacts."""
        assert len(new_assembly.connections) > 0, "No connections found by new assembly"
        print(f"\nNew assembly found {len(new_assembly.connections)} connections")
    
    @pytest.mark.slow
    def test_connection_count_similar(self, reference_connections, new_assembly):
        """Test that the number of connections is roughly similar."""
        ref_count = len(reference_connections)
        new_count = len(new_assembly.connections)
        
        # Allow some tolerance (within 50% of reference)
        min_expected = int(ref_count * 0.5)
        max_expected = int(ref_count * 1.5)
        
        print(f"\nReference connections: {ref_count}")
        print(f"New assembly connections: {new_count}")
        
        assert min_expected <= new_count <= max_expected, (
            f"Connection count {new_count} is outside expected range [{min_expected}, {max_expected}]"
        )
    
    @pytest.mark.slow
    def test_connection_pairs_overlap(self, reference_connections, new_assembly):
        """Test that at least some connection pairs match between old and new."""
        # Get part pairs from new assembly
        new_pairs = set()
        for conn in new_assembly.connections:
            if isinstance(conn, Geometric_Connection):
                key = tuple(sorted([conn.part_name1, conn.part_name2]))
                new_pairs.add(key)
        
        # Get part pairs from reference
        ref_pairs = set(reference_connections.keys())
        
        # Calculate overlap
        common_pairs = new_pairs & ref_pairs
        
        print(f"\nReference pairs: {len(ref_pairs)}")
        print(f"New pairs: {len(new_pairs)}")
        print(f"Common pairs: {len(common_pairs)}")
        
        # At least 50% of reference pairs should be found
        overlap_ratio = len(common_pairs) / len(ref_pairs) if ref_pairs else 0
        assert overlap_ratio >= 0.5, (
            f"Only {overlap_ratio*100:.1f}% overlap in connection pairs (expected >= 50%)"
        )
    
    @pytest.mark.slow
    def test_contact_points_in_similar_regions(self, reference_connections, new_assembly):
        """
        Test that contact points from new assembly are in similar regions
        as the reference contact points.
        """
        mismatches = []
        matches = 0
        
        for conn in new_assembly.connections:
            if not isinstance(conn, Geometric_Connection):
                continue
                
            key = tuple(sorted([conn.part_name1, conn.part_name2]))
            if key not in reference_connections:
                continue
            
            ref_data = reference_connections[key]
            ref_points = ref_data["points"]
            new_points = np.asarray(conn.points)
            
            if len(ref_points) == 0 or len(new_points) == 0:
                continue
            
            # Compare bounding boxes of contact points
            ref_min = np.min(ref_points, axis=0)
            ref_max = np.max(ref_points, axis=0)
            new_min = np.min(new_points, axis=0)
            new_max = np.max(new_points, axis=0)
            
            # Check if bounding boxes overlap (with some tolerance)
            tolerance = 5.0  # mm tolerance
            ref_min_expanded = ref_min - tolerance
            ref_max_expanded = ref_max + tolerance
            
            # Check overlap
            overlaps = np.all(new_min <= ref_max_expanded) and np.all(new_max >= ref_min_expanded)
            
            if overlaps:
                matches += 1
            else:
                mismatches.append((conn.part_name1, conn.part_name2))
        
        print(f"\nContact region matches: {matches}")
        print(f"Contact region mismatches: {len(mismatches)}")
        
        if mismatches:
            print(f"Mismatched pairs: {mismatches[:5]}...")  # Show first 5
        
        # At least 70% of common connections should have overlapping contact regions
        total_checked = matches + len(mismatches)
        if total_checked > 0:
            match_ratio = matches / total_checked
            assert match_ratio >= 0.7, (
                f"Only {match_ratio*100:.1f}% of connections have overlapping contact regions"
            )


class TestWingAssemblyContactDetection:
    """Tests for individual contact detection functionality."""
    
    @pytest.mark.slow
    def test_find_contacts_on_subset(self):
        """Test contact detection on a small subset of parts."""
        # Load just 3 parts that should have contacts
        part_names = ["CMT5730B072-041", "CMT5730B074-041", "CMT5730B078-101"]
        parts = []
        
        for name in part_names:
            stl_path = os.path.join(STL_DIR, f"{name}.stl")
            if os.path.exists(stl_path):
                part = Part(STL_TO_PART_NAME.get(name, name), stl_path)
                parts.append(part)
        
        if len(parts) < 2:
            pytest.skip("Not enough parts found for subset test")
        
        assembly = Assembly("subset_test", parts, contact_tollerance=1.0)
        assembly.find_contacts()
        
        print(f"\nSubset assembly with {len(parts)} parts found {len(assembly.connections)} connections")
        
        # Should find at least some connections
        assert len(assembly.connections) > 0, "No connections found in subset"
    
    @pytest.mark.slow
    def test_contact_points_have_valid_normals(self, new_assembly):
        """Test that all contact points have valid (non-zero) normals."""
        for conn in new_assembly.connections:
            if isinstance(conn, Geometric_Connection) and conn.normals is not None:
                normals = np.asarray(conn.normals)
                if len(normals) > 0:
                    # Check normals are roughly unit length
                    norms = np.linalg.norm(normals, axis=1)
                    valid_norms = np.all((norms > 0.9) & (norms < 1.1))
                    assert valid_norms, (
                        f"Connection {conn.part_name1}-{conn.part_name2} has invalid normals"
                    )


class TestWingAssemblyHelperFunctions:
    """Tests for helper functions used in contact detection."""
    
    def test_o3d_offset_mesh(self):
        """Test that mesh offset function works correctly."""
        from diss_assembly_dataclasses.assembly import o3d_offset_mesh
        import open3d as o3d
        
        # Create a simple cube mesh
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        mesh.compute_vertex_normals()
        
        original_center = mesh.get_center()
        
        # Offset outward
        offset_mesh = o3d_offset_mesh(mesh, 0.1)
        
        # The offset mesh should have the same number of vertices/triangles
        assert len(offset_mesh.vertices) == len(mesh.vertices)
        assert len(offset_mesh.triangles) == len(mesh.triangles)
        
        # The bounding box should be larger
        orig_bb = mesh.get_axis_aligned_bounding_box()
        offset_bb = offset_mesh.get_axis_aligned_bounding_box()
        
        # Volume of offset bounding box should be larger
        orig_extent = np.asarray(orig_bb.get_extent())
        offset_extent = np.asarray(offset_bb.get_extent())
        
        assert np.all(offset_extent >= orig_extent), "Offset mesh should be larger"
    
    def test_o3d_get_triangle_centers(self):
        """Test triangle center computation."""
        from diss_assembly_dataclasses.assembly import o3d_get_triangle_centers
        import open3d as o3d
        
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        centers = o3d_get_triangle_centers(mesh)
        
        assert len(centers) == len(mesh.triangles)
        assert centers.shape[1] == 3
        
        # Centers should be within the bounding box
        bb = mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bb.min_bound)
        max_bound = np.asarray(bb.max_bound)
        
        assert np.all(centers >= min_bound - 0.01)
        assert np.all(centers <= max_bound + 0.01)
    
    def test_fcl_check_collision_touching(self):
        """Test FCL collision detection with touching meshes."""
        from diss_assembly_dataclasses.assembly import fcl_check_collision
        import open3d as o3d
        
        # Create two boxes that touch
        box1 = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        box2 = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        box2.translate([0.9, 0, 0])  # Overlapping slightly
        
        collision, contacts = fcl_check_collision(box1, box2)
        
        assert collision, "Overlapping boxes should collide"
    
    def test_fcl_check_collision_separate(self):
        """Test FCL collision detection with separate meshes."""
        from diss_assembly_dataclasses.assembly import fcl_check_collision
        import open3d as o3d
        
        # Create two boxes that don't touch
        box1 = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        box2 = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        box2.translate([5.0, 0, 0])  # Far apart
        
        collision, contacts = fcl_check_collision(box1, box2)
        
        assert not collision, "Separate boxes should not collide"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
