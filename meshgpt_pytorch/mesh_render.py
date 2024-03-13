def combind_mesh(path, mesh):
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    translation_distance = 0.5  

    for r, faces_coordinates in enumerate(mesh):  
        numpy_data = faces_coordinates[0].cpu().numpy().reshape(-1, 3)   
        numpy_data[:, 0] += translation_distance * (r / 0.2 - 1)  
    
        for vertex in numpy_data:
            all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    
        for i in range(1, len(numpy_data), 3):
            all_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")
    
        vertex_offset += len(numpy_data)
    
    obj_file_content = "".join(all_vertices) + "".join(all_faces)
     
    with open(path , "w") as file:
        file.write(obj_file_content)   
 
def combind_mesh_with_rows(path, meshes):
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    translation_distance = 0.5  
    obj_file_content = ""
    
    for row, mesh in enumerate(meshes): 
        for r, faces_coordinates in enumerate(mesh): 
            numpy_data = faces_coordinates[0].cpu().numpy().reshape(-1, 3)  
            numpy_data[:, 0] += translation_distance * (r / 0.2 - 1)  
            numpy_data[:, 2] += translation_distance * (row / 0.2 - 1)  
        
            for vertex in numpy_data:
                all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
            for i in range(1, len(numpy_data), 3):
                all_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")
        
            vertex_offset += len(numpy_data)
        
        obj_file_content = "".join(all_vertices) + "".join(all_faces)
     
    with open(path , "w") as file:
        file.write(obj_file_content)   
        
        
def write_mesh_output(path, coords):
    numpy_data = coords[0].cpu().numpy().reshape(-1, 3)  
    obj_file_content = ""
    
    for vertex in numpy_data:
        obj_file_content += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"

    for i in range(1, len(numpy_data), 3):
        obj_file_content += f"f {i} {i + 1} {i + 2}\n"
 
    with open(path, "w") as file:
        file.write(obj_file_content) 
         