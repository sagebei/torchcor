import torch


class Matrices1D_3DSurface:
    def __init__(self, vertices, elems, device, dtype):
        self.elems = elems
        self.matrices_1d = Matrices1D(vertices, elems.Ln.data, device, dtype)
        self.matrices_3d = Matrices3DSurface(vertices, elems.Tr.data, device, dtype)

    def assemble_matrices(self, sigma):
        K_1d, M_1d = self.matrices_1d.assemble_matrices(sigma[self.elems.Ln.idx])
        K_3d, M_3d = self.matrices_3d.assemble_matrices(sigma[self.elems.Tr.idx])

        K = K_1d + K_3d
        M = M_1d + M_3d

        return K.coalesce(), M.coalesce()


class Matrices1D:
    def __init__(self, vertices, lines, device, dtype):
        """
        vertices: (N, 2)
        lines:    (M, 2)
        """
        self.vertices = vertices.clone()
        self.n_vertices = vertices.shape[0]
        self.lines = lines.clone()
        self.device = device
        self.dtype = dtype

    def shape_function_gradients(self):
        return torch.tensor([[-1.0],
                             [ 1.0]],
                             device=self.device,
                             dtype=self.dtype)

    def jacobian(self, line_coords):
        # line_coords: (M,2,d)
        dN_dxi = self.shape_function_gradients()  # (2,1)

        # (M,d,2) @ (2,1) → (M,d,1)
        J = line_coords.transpose(1, 2) @ dN_dxi
        return J  # (M,d,1)

    def calculate_measure(self, J_batch):
        # |J| = element length
        return torch.linalg.norm(J_batch.squeeze(-1), dim=1)

    def local_mass(self, lengths):
        Me_template = torch.tensor([[2.0, 1.0],
                                    [1.0, 2.0]],
                                    device=self.device,
                                    dtype=self.dtype)

        return (lengths / 6).unsqueeze(1).unsqueeze(2) * Me_template.unsqueeze(0)

    def local_stiffness(self, sigma, line_coords):
        J_batch = self.jacobian(line_coords)          # (M,d,1)
        lengths = self.calculate_measure(J_batch)    # (M,)

        # Inverse Jacobian
        # For 1D: J is (M,d,1). We treat mapping scalar-wise.
        # J^T J = |J|^2 so inverse in least squares sense:
        JT = J_batch.transpose(1, 2)                 # (M,1,d)
        inv_J = JT / (lengths**2).view(-1,1,1)       # (M,1,d)

        dN_dxi = self.shape_function_gradients()     # (2,1)

        # (2,1) @ (M,1,d) → (M,2,d)
        dN_dx = dN_dxi @ inv_J

        # K_e = |J| * ∇N sigma ∇N^T
        Ke_batch = lengths.view(-1,1,1) * \
                   (dN_dx @ (sigma @ dN_dx.transpose(1,2)))

        return Ke_batch

    def construct_local_matrices(self, sigma):
        self.vertices = self.vertices.to(self.device)
        self.lines = self.lines.to(self.device)

        line_coords = self.vertices[self.lines]  # (M,2,d)

        J_batch = self.jacobian(line_coords)
        lengths = self.calculate_measure(J_batch)

        Me_batch = self.local_mass(lengths)
        Ke_batch = self.local_stiffness(sigma, line_coords)

        return Me_batch, Ke_batch

    def assemble_matrices(self, sigma):
        Me_batch, Ke_batch = self.construct_local_matrices(sigma)

        rows, cols, K_vals, M_vals = [], [], [], []

        for i in range(2):
            for j in range(2):
                rows.extend(self.lines[:, i].tolist())
                cols.extend(self.lines[:, j].tolist())
                K_vals.extend(Ke_batch[:, i, j].tolist())
                M_vals.extend(Me_batch[:, i, j].tolist())

        K = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=K_vals,
            size=(self.n_vertices, self.n_vertices),
            device=self.device,
            dtype=self.dtype
        ).coalesce()

        M = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=M_vals,
            size=(self.n_vertices, self.n_vertices),
            device=self.device,
            dtype=self.dtype
        ).coalesce()

        return K, M



class Matrices2D:
    def __init__(self, vertices, triangles, device, dtype):
        self.vertices = vertices.clone()  # (12100, 2)
        self.n_vertices = vertices.shape[0]
        self.triangles = triangles.clone()  # (23762, 3)
        self.device = device
        self.dtype = dtype

    def shape_function_gradients(self):
        # Gradients of shape functions in the reference element (master triangle) (3, 2)
        return torch.tensor([[-1, -1],
                                  [1, 0],
                                  [0, 1]], device=self.device, dtype=self.dtype)

    def jacobian(self, triangle_coords):
        dN_dxi = self.shape_function_gradients()

        # Transpose `coords` (N, 3, 2) so that it becomes (N, 2, 3) to match (3, 2) of dN_dxi
        J = triangle_coords.transpose(1, 2) @ dN_dxi  # Result: (N, 2, 2)

        return J

    def local_mass(self, areas):
        Me_batch = (areas / 12).unsqueeze(1).unsqueeze(2) * torch.tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]],
                                                                         device=self.device,
                                                                         dtype=self.dtype).unsqueeze(0)  # Shape (N, 3, 3)
        return Me_batch

    def local_stiffness(self, sigma, triangle_coords):
        # Calculate the Jacobian for each triangle
        J_batch = self.jacobian(triangle_coords)  
        det_J_batch = torch.linalg.det(J_batch)
        inv_J_batch = torch.linalg.inv(J_batch)

        # Calculate dN/dxy for all triangles using the inverse Jacobian
        dN_dxi = self.shape_function_gradients()  # Assuming this is constant for all triangles, shape (3, 2)
        dN_dxy_batch = dN_dxi @ inv_J_batch  # Shape (N, 3, 2)
        Ke_batch = (0.5 * det_J_batch).view(-1, 1, 1) * dN_dxy_batch @ (sigma @ dN_dxy_batch.transpose(1, 2))

        return Ke_batch
    
    def calculate_area(self, triangle_coords):
        coords_augmented = torch.cat([torch.ones(triangle_coords.shape[0], 3, 1, device=self.device, dtype=self.dtype), triangle_coords], dim=2)
        areas = 0.5 * torch.abs(torch.linalg.det(coords_augmented))

        return areas
    
    def construct_local_matrices(self, sigma):
        self.vertices = self.vertices.to(self.device)
        self.triangles = self.triangles.to(self.device)

        triangle_coords = self.vertices[self.triangles]  # (23762, 3, 2)

        # Calculate areas for all triangles at once
        areas = self.calculate_area(triangle_coords)

        # Mass and stiffness matrix for all triangles: (N, 3, 3)
        Me_batch = self.local_mass(areas)
        Ke_batch = self.local_stiffness(sigma, triangle_coords)

        return Me_batch, Ke_batch

    def assemble_matrices(self, sigma):
        # Precompute vertex coordinates for all triangles
        Me_batch, Ke_batch = self.construct_local_matrices(sigma)
        # Lists to store the indices and values of non-zero elements for K and M
        rows, cols, K_vals, M_vals = [], [], [], []
        # Collect contributions for global matrices in sparse form
        for i in range(3):
            for j in range(3):
                rows.extend(self.triangles[:, i].tolist())
                cols.extend(self.triangles[:, j].tolist())
                K_vals.extend(Ke_batch[:, i, j].tolist())
                M_vals.extend(Me_batch[:, i, j].tolist())

        # Create sparse tensors from the accumulated lists
        K = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=K_vals,
            size=(self.n_vertices, self.n_vertices)
        )

        M = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=M_vals,
            size=(self.n_vertices, self.n_vertices)
        )

        K = K.to(device=self.device, dtype=self.dtype)
        M = M.to(device=self.device, dtype=self.dtype)

        return K.coalesce(), M.coalesce()
    

class Matrices3DSurface(Matrices2D):
    def __init__(self, vertices, triangles, device, dtype):
        super().__init__(vertices, triangles, device, dtype)

    def rotation_matrix(self, triangle_coords):
        u1 = triangle_coords[:, 1, :] - triangle_coords[:, 0, :]  # (N, 3)
        u2 = triangle_coords[:, 2, :] - triangle_coords[:, 0, :]  # (N, 3)

        alpha = u1 / torch.linalg.norm(u1, dim=1, keepdim=True)  # (N, 3)

        beta = u2 - torch.sum(u2 * alpha, dim=1, keepdim=True) * alpha  # (N, 3)
        beta = beta / torch.linalg.norm(beta, dim=1, keepdim=True)  # (N, 3)

        rotation_matrix = torch.stack([alpha, beta], dim=1)  # (N, 2, 3)
        return rotation_matrix

    def jacobian(self, triangle_coords):
        dN_dxi = self.shape_function_gradients()
        # Transpose `coords` to match dimensions: (N, 3, 3)
        rotation_matrix = self.rotation_matrix(triangle_coords)
        J = rotation_matrix @ triangle_coords.transpose(1, 2) @ dN_dxi

        return J
    
    def calculate_area(self, triangle_coords):
        # Calculate the area of the triangle using cross product
        a = triangle_coords[:, 1] - triangle_coords[:, 0]
        b = triangle_coords[:, 2] - triangle_coords[:, 0]
        cross_product = torch.linalg.cross(a, b)  # Shape (N, 3)
        areas = 0.5 * torch.norm(cross_product, dim=1)  # Shape (N,)

        return areas

    def local_stiffness(self, sigma, triangle_coords):
        rotation_matrix = self.rotation_matrix(triangle_coords)  # (N, 2, 3)
        sigma = rotation_matrix @ sigma @ rotation_matrix.transpose(1, 2)

        J_batch = self.jacobian(triangle_coords)
        det_J_batch = torch.linalg.det(J_batch)
        inv_J_batch = torch.linalg.inv(J_batch)

        # Calculate dN/dxy for all triangles using the inverse Jacobian
        dN_dxi = self.shape_function_gradients()  # Assuming this is constant for all triangles, shape (3, 2)
        dN_dxy_batch = dN_dxi @ inv_J_batch  # Shape (N, 3, 2)
        Ke_batch = (0.5 * det_J_batch).view(-1, 1, 1) * dN_dxy_batch @ (sigma @ dN_dxy_batch.transpose(1, 2))

        return Ke_batch


class Matrices3D:
    def __init__(self, vertices, tetrahedrons, device, dtype):
        self.vertices = vertices.clone()  # (N, 3) for 3D vertices
        self.n_vertices = vertices.shape[0]
        self.tetrahedrons = tetrahedrons.clone()  # (M, 4) for 4-node tetrahedrons
        self.device = device
        self.dtype = dtype

    def shape_function_gradients(self):
        # Gradients of shape functions in the reference element (master tetrahedron) (4, 3)
        return torch.tensor([[-1, -1, -1],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], device=self.device, dtype=self.dtype)

    def jacobian(self, tetrahedron_coords):
        dN_dxi = self.shape_function_gradients()  # (4, 3)
        
        # Transpose `coords` (M, 4, 3) to (M, 3, 4) for matrix multiplication
        J = tetrahedron_coords.transpose(1, 2) @ dN_dxi  # Result: (M, 3, 3) for each tetrahedron

        return J

    def local_mass(self, volumes):
        Me_batch = (volumes / 20).unsqueeze(1).unsqueeze(2) * torch.tensor([[2, 1, 1, 1],
                                                                            [1, 2, 1, 1],
                                                                            [1, 1, 2, 1],
                                                                            [1, 1, 1, 2]],
                                                                            device=self.device, 
                                                                            dtype=self.dtype).unsqueeze(0)  # Shape (M, 4, 4)
        return Me_batch

    def local_stiffness(self, sigma, tetrahedron_coords):
        # Calculate the Jacobian for each tetrahedron
        J_batch = self.jacobian(tetrahedron_coords)  # (M, 3, 3)
        det_J_batch = torch.abs(torch.linalg.det(J_batch))

        inv_J_batch = torch.linalg.inv(J_batch)

        # Calculate dN/dxyz for all tetrahedrons using the inverse Jacobian
        dN_dxi = self.shape_function_gradients()  # Shape (4, 3)
        dN_dxyz_batch = dN_dxi @ inv_J_batch  # Shape (M, 4, 3)
        Ke_batch = (det_J_batch / 6).view(-1, 1, 1) * (dN_dxyz_batch @ (sigma @ dN_dxyz_batch.transpose(1, 2)))

        return Ke_batch
    
    def calculate_volume(self, tetrahedron_coords):
        # Adding a column of ones to calculate the volume determinant
        coords_augmented = torch.cat([torch.ones(tetrahedron_coords.shape[0], 4, 1, device=self.device, dtype=self.dtype),
                                      tetrahedron_coords], dim=2)
        volumes = torch.abs(torch.linalg.det(coords_augmented)) / 6

        return volumes

    def construct_local_matrices(self, sigma):
        self.vertices = self.vertices.to(self.device)
        self.tetrahedrons = self.tetrahedrons.to(self.device)

        tetrahedron_coords = self.vertices[self.tetrahedrons]  # (M, 4, 3)

        # Calculate volumes for all tetrahedrons at once
        volumes = self.calculate_volume(tetrahedron_coords)

        # Mass and stiffness matrices for all tetrahedrons: (M, 4, 4)
        Me_batch = self.local_mass(volumes)
        Ke_batch = self.local_stiffness(sigma, tetrahedron_coords)

        return Me_batch, Ke_batch

    def assemble_matrices(self, sigma):
        # Precompute vertex coordinates for all tetrahedrons
        Me_batch, Ke_batch = self.construct_local_matrices(sigma)
        
        # Lists to store the indices and values of non-zero elements for K and M
        rows, cols, K_vals, M_vals = [], [], [], []
        
        # Collect contributions for global matrices in sparse form
        for i in range(4):
            for j in range(4):
                rows.extend(self.tetrahedrons[:, i].tolist())
                cols.extend(self.tetrahedrons[:, j].tolist())
                K_vals.extend(Ke_batch[:, i, j].tolist())
                M_vals.extend(Me_batch[:, i, j].tolist())

        # Create sparse tensors from the accumulated lists
        K = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=K_vals,
            size=(self.n_vertices, self.n_vertices)
        )

        M = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=M_vals,
            size=(self.n_vertices, self.n_vertices)
        )

        K = K.to(device=self.device, dtype=self.dtype)
        M = M.to(device=self.device, dtype=self.dtype)

        return K.coalesce(), M.coalesce()
