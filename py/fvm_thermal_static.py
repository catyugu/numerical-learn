import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

# ==========================================
# 1. MESH GENERATION (Unstructured)
# ==========================================
def generate_mesh(n_points=1500):
    # Generate random points in a 1x1 box
    points = np.random.rand(n_points, 2)
    
    # Define a hole in the center (Circle radius 0.25 at 0.5, 0.5)
    center = np.array([0.5, 0.5])
    radius = 0.25
    dist = np.linalg.norm(points - center, axis=1)
    points = points[dist > radius]

    # Add explicit boundary points to ensure the shape is preserved
    theta = np.linspace(0, 2*np.pi, 100)
    hole_bound = np.column_stack([0.5 + radius*np.cos(theta), 0.5 + radius*np.sin(theta)])
    
    # Outer box boundary
    box_x = np.linspace(0, 1, 25)
    box_y = np.linspace(0, 1, 25)
    box_bound = np.vstack([
        np.column_stack([box_x, np.zeros_like(box_x)]), # Bottom
        np.column_stack([box_x, np.ones_like(box_x)]),  # Top
        np.column_stack([np.zeros_like(box_y), box_y]), # Left
        np.column_stack([np.ones_like(box_y), box_y])   # Right
    ])
    
    # Combine and triangulate
    all_points = np.vstack([points, hole_bound, box_bound])
    tri = Delaunay(all_points)
    
    # Filter out triangles that might have formed *inside* the hole 
    centroids = np.mean(all_points[tri.simplices], axis=1)
    dist_c = np.linalg.norm(centroids - center, axis=1)
    keep = dist_c > radius
    simplices = tri.simplices[keep]
    
    return all_points, simplices

# ==========================================
# 2. PRE-PROCESSING (Topology)
# ==========================================
def get_face_connectivity(points, simplices):
    """
    Converts cell-based data (triangles) into face-based data.
    Returns: 
       faces: list of edges (node_a, node_b)
       face_to_cells: map of face_index -> [owner_cell, neighbor_cell]
       face_centers: coordinates of face midpoints
       face_normals: normal vectors for each face
       face_lengths: length of each face
    """
    edges = {} # Key: (min_node, max_node), Value: list of cell_indices
    
    for cell_idx, simplex in enumerate(simplices):
        # A triangle has 3 faces (edges)
        for i in range(3):
            p1, p2 = simplex[i], simplex[(i+1)%3]
            edge_key = tuple(sorted((p1, p2)))
            
            if edge_key not in edges:
                edges[edge_key] = []
            edges[edge_key].append(cell_idx)
            
    # Process edges into arrays
    face_list = []
    face_owners = []
    face_neighbors = []
    
    for edge, cells in edges.items():
        face_list.append(edge)
        face_owners.append(cells[0])
        # If an edge has only 1 cell, it is a Boundary Face. We mark neighbor as -1.
        face_neighbors.append(cells[1] if len(cells) > 1 else -1)
        
    face_owners = np.array(face_owners)
    face_neighbors = np.array(face_neighbors)
    
    # Calculate geometry
    p1 = points[np.array([f[0] for f in face_list])]
    p2 = points[np.array([f[1] for f in face_list])]
    
    face_centers = 0.5 * (p1 + p2)
    face_deltas = p2 - p1
    face_lengths = np.linalg.norm(face_deltas, axis=1)
    
    # Normal vector: rotate edge vector 90 degrees (dx, dy) -> (dy, -dx)
    # Note: Direction matters (pointing out of owner vs neighbor), handled in loop later
    face_normals = np.column_stack([face_deltas[:,1], -face_deltas[:,0]]) 
    face_normals /= face_lengths[:, None] # Normalize
    
    return face_owners, face_neighbors, face_centers, face_lengths, face_normals

# ==========================================
# 3. FVM SOLVER
# ==========================================
def solve_thermal_fvm():
    # Parameters
    k_cond = 10.0        # Thermal conductivity
    T_cold = 20.0       # Dirichlet val
    h_conv = 5.0       # Convection Coeff (Robin)
    T_inf = 60.0        # Ambient Temp (Robin)
    
    # 1. Mesh
    points, simplices = generate_mesh(2400)
    n_cells = len(simplices)
    
    # Calculate Cell Centers (centroids)
    cell_centers = np.mean(points[simplices], axis=1)
    
    # 2. Topology
    owners, neighbors, f_centers, f_lengths, f_normals = get_face_connectivity(points, simplices)
    n_faces = len(owners)
    
    # 3. Matrix Assembly (A * T = b)
    A = lil_matrix((n_cells, n_cells))
    b = np.zeros(n_cells)
    
    print(f"Mesh: {n_cells} cells, {n_faces} faces.")
    
    # Loop over all faces
    for f in range(n_faces):
        owner = owners[f]
        neigh = neighbors[f]
        
        # Centroid of owner
        c_owner = cell_centers[owner]
        
        # --- INTERNAL FACES ---
        if neigh != -1:
            c_neigh = cell_centers[neigh]
            
            # Distance between cell centers (d_ON)
            dist_vec = c_neigh - c_owner
            dist = np.linalg.norm(dist_vec)
            
            # Correction: Ensure normal points from Owner -> Neighbor
            # Dot product of dist vector and normal
            if np.dot(dist_vec, f_normals[f]) < 0:
                 # Normal is flipped relative to Owner->Neighbor direction
                 # In full CFD we fix the normal, here we just use abs for diffusion
                 pass 

            # Transmissibility (k * Area / Distance)
            # Area in 2D is the edge length
            a_f = k_cond * f_lengths[f] / dist
            
            # Add to matrix (Flux leaving owner, entering neighbor)
            # Owner Equation
            A[owner, owner] += a_f
            A[owner, neigh] -= a_f
            
            # Neighbor Equation
            A[neigh, neigh] += a_f
            A[neigh, owner] -= a_f
            
        # --- BOUNDARY FACES ---
        else:
            # We need to determine WHICH boundary this face belongs to.
            # We use the coordinates of the face center.
            xf, yf = f_centers[f]
            dist_from_hole = np.sqrt((xf-0.5)**2 + (yf-0.5)**2)
            
            # Distance from cell center to face center
            d_Pf = np.linalg.norm(f_centers[f] - c_owner)
            
            # -- BC 1: LEFT WALL (Dirichlet: T = T_cold) --
            if xf < 0.01: 
                # Flux = k * A * (T_face - T_cell) / d_Pf
                # We know T_face = T_cold.
                coeff = k_cond * f_lengths[f] / d_Pf
                
                A[owner, owner] += coeff
                b[owner] += coeff * T_cold
                
            # -- BC 2: RIGHT WALL (Neumann: Adiabatic/Insulated) --
            elif xf > 0.99:
                # Flux = 0. Do nothing. 
                # (Implicitly Neumann zero gradient in FVM)
                pass
                
            # -- BC 3: HOLE (Robin: Convection) --
            elif dist_from_hole < 0.3: # Check if close to hole center
                # Flux = h * A * (T_cell - T_inf)
                # Note: Heat leaves cell if T_cell > T_inf
                
                coeff = h_conv * f_lengths[f]
                
                A[owner, owner] += coeff  # Implicit part of T_cell
                b[owner] += coeff * T_inf # Explicit source part
            
            # -- Top/Bottom Walls (Let's make them Adiabatic too for simplicity) --
            else:
                pass

    # 4. Solve
    print("Solving linear system...")
    T = spsolve(A.tocsr(), b)
    
    # 5. Visualize
    print("Visualizing...")
    plt.figure(figsize=(10, 8))
    
    # Method: Color the triangles based on the calculated T (Cell centered)
    triang = mtri.Triangulation(points[:,0], points[:,1], simplices)
    
    plt.tripcolor(triang, facecolors=T, cmap='inferno', edgecolors='k', lw=0.1)
    cbar = plt.colorbar()
    cbar.set_label('Temperature (T)')
    
    plt.title("FVM on Unstructured Grid: Plate with Hole")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    solve_thermal_fvm()