import numpy as np
import math

def orient_triangle_upward(v1, v2, v3): 
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2) 
    normal = normal / np.linalg.norm(normal)
     
    up = np.array([0, 1, 0])
    if np.dot(normal, up) < 0: 
        v1, v3 = v3, v1 
    return v1, v2, v3

def get_angle(v1, v2, v3):
    v1, v2, v3 = orient_triangle_upward(v1, v2, v3) 
    vec1 = v2 - v1
    vec2 = v3 - v1 
    angle_rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return math.degrees(angle_rad)  
 
def save_rendering(path, input_meshes):
    all_vertices,all_faces = [],[]
    vertex_offset = 0
    translation_distance = 0.5  
    obj_file_content = ""
    meshes = input_meshes if isinstance(input_meshes, list) else [input_meshes] 
    
    for row, mesh in enumerate(meshes): 
        mesh = mesh if isinstance(mesh, list) else [mesh]
        cell_offset = 0
        for tensor, mask in mesh:   
            for tensor_batch, mask_batch in zip(tensor,mask):  
                numpy_data = tensor_batch[mask_batch].cpu().numpy().reshape(-1, 3)  
                numpy_data[:, 0] += translation_distance * (cell_offset / 0.2 - 1)  
                numpy_data[:, 2] += translation_distance * (row / 0.2 - 1)  
                cell_offset += 1
                for vertex in numpy_data:
                    all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                
                mesh_center = np.mean(numpy_data, axis=0)
                for i in range(0, len(numpy_data), 3):
                    v1 = numpy_data[i]
                    v2 = numpy_data[i + 1]
                    v3 = numpy_data[i + 2]
  
                    normal = np.cross(v2 - v1, v3 - v1)   
                    if get_angle(v1, v2, v3) > 60:  
                        direction_vector = mesh_center - np.mean([v1, v2, v3], axis=0)
                        direction_vector = -direction_vector
                    else:
                        direction_vector = [0, 1, 0]  
                         
                    if np.dot(normal, direction_vector) > 0: 
                        order =  [0, 1, 2]
                    else:
                        order = [0, 2, 1]

                    reordered_vertices = [v1, v2, v3][order[0]], [v1, v2, v3][order[1]], [v1, v2, v3][order[2]] 
                    indices = [np.where((numpy_data == vertex).all(axis=1))[0][0] + 1 + vertex_offset for vertex in reordered_vertices] 
                    all_faces.append(f"f {indices[0]} {indices[1]} {indices[2]}\n")

                vertex_offset += len(numpy_data)   
        obj_file_content = "".join(all_vertices) + "".join(all_faces)
     
    with open(path , "w") as file:
        file.write(obj_file_content) 
        
    print(f"[Save_rendering] Saved at {path}") 
         