# %%
from numpy import array, zeros, add, subtract, matmul, insert, cross, divide
from numpy import linspace, vstack, hstack, allclose, where, full, searchsorted, radians, sin, cos
from numpy import ndarray
from numpy.linalg import inv, pinv
from math import isclose
from Pynite.BeamSegZ import BeamSegZ
from Pynite.BeamSegY import BeamSegY
import Pynite.FixedEndReactions
import warnings

#%%
class Member3D():
    """
    A class representing a 3D frame element in a finite element model.

    Most users will not need to interface with this class directly. Rather, the physical member
    class, which inherits from this class and stitches together a seires of colinear `Member3D`
    objects will be more useful.
    """

    # '__plt' is used to store the 'pyplot' from matplotlib once it gets imported. Setting it to 'None' for now allows us to defer importing it until it's actually needed.
    __plt = None

#%%
    def __init__(self, model, name, i_node, j_node, material_name, section_name, rotation=0.0,
                 tension_only=False, comp_only=False):
        """
        Initializes a new member.
        """
        self.name = name      # A unique name for the member given by the user
        self.ID = None        # Unique index number for the member assigned by the program
        self.i_node = i_node  # The element's i-node
        self.j_node = j_node  # The element's j-node

        try:
            self.material = model.materials[material_name] # The element's material
        except KeyError:
            raise NameError(f"No material named '{material_name}'")
        
        try:
            self.section = model.sections[section_name] # The element's section
        except KeyError:
            raise NameError(f"No section names '{section_name}'")
        
        # Variables used to track nonlinear material member end forces
        self._fxi = 0
        self._myi = 0
        self._mzi= 0
        self._fxj = 0
        self._myj = 0
        self._mzj = 0

        # Variable used to track plastic load reveral
        self.i_reversal = False
        self.j_reversal = False

        self.rotation = rotation  # Member rotation (degrees) about its local x-axis
        self.PtLoads = []         # A list of point loads & moments applied to the element (Direction, P, x, case='Case 1') or (Direction, M, x, case='Case 1')
        self.DistLoads = []       # A list of linear distributed loads applied to the element (Direction, w1, w2, x1, x2, case='Case 1')
        self.SegmentsZ = []       # A list of mathematically continuous beam segments for z-bending
        self.SegmentsY = []       # A list of mathematically continuous beam segments for y-bending
        self.SegmentsX = []       # A list of mathematically continuous beam segments for torsion
        self.Releases = [False, False, False, False, False, False, False, False, False, False, False, False]
        self.tension_only = tension_only # Indicates whether the member is tension-only
        self.comp_only = comp_only # Indicates whether the member is compression-only

        # Members need to track whether they are active or not for any given load combination. They may become inactive for a load combination during a tension/compression-only analysis. This dictionary will be used when the model is solved.
        self.active = {} # Key = load combo name, Value = True or False
        
        # The 'Member3D' object will store results for one load combination at a time. To reduce repetative calculations the '_solved_combo' variable will be used to track whether the member needs to be resegmented before running calculations for any given load combination.
        self._solved_combo = None # The current solved load combination

        # Members need a link to the model they belong to
        self.model = model

#%%
    def L(self):
        """
        Returns the length of the member.
        """

        # Return the distance between the two nodes
        return self.i_node.distance(self.j_node)

#%%
    def _partition_D(self):
        """
        Builds lists of unreleased and released degree of freedom indices for the member.

        Returns
        -------
        R1_indices : list
            A list of the indices for the unreleased DOFs
        R2_indices : list
            A list of the indices for the released DOFs
        """

        R1_indices = []
        R2_indices = []
        for i in range(12):
            if self.Releases[i] == False:
                R1_indices.append(i)
            else:
                R2_indices.append(i)
        
        return R1_indices, R2_indices

#%%
    def k(self):
        """
        Returns the condensed (and expanded) local stiffness matrix for the member.
        """

        # Partition the local stiffness matrix as 4 submatrices in
        # preparation for static condensation
        k11, k12, k21, k22 = self._partition(self._k_unc())
               
        # Calculate the condensed local stiffness matrix
        k_Condensed = subtract(k11, matmul(matmul(k12, inv(k22)), k21))
        
        # Expand the condensed local stiffness matrix
        i=0
        for DOF in self.Releases:
            
            if DOF == True:
                k_Condensed = insert(k_Condensed, i, 0, axis = 0)
                k_Condensed = insert(k_Condensed, i, 0, axis = 1)
                
            i += 1

        # Return the local stiffness matrix, with end releases applied
        return k_Condensed

#%%
    def _k_unc(self):
        """
        Returns the uncondensed local stiffness matrix for the member.
        """

        # Get the properties needed to form the local stiffness matrix
        E = self.material.E
        G = self.material.G
        Iy = self.section.Iy
        Iz = self.section.Iz
        J = self.section.J
        A = self.section.A
        L = self.L()
        
        # Create the uncondensed local stiffness matrix
        k = array([[A*E/L,  0,             0,             0,      0,            0,            -A*E/L, 0,             0,             0,      0,            0],
                    [0,      12*E*Iz/L**3,  0,             0,      0,            6*E*Iz/L**2,  0,      -12*E*Iz/L**3, 0,             0,      0,            6*E*Iz/L**2],
                    [0,      0,             12*E*Iy/L**3,  0,      -6*E*Iy/L**2, 0,            0,      0,             -12*E*Iy/L**3, 0,      -6*E*Iy/L**2, 0],
                    [0,      0,             0,             G*J/L,  0,            0,            0,      0,             0,             -G*J/L, 0,            0],
                    [0,      0,             -6*E*Iy/L**2,  0,      4*E*Iy/L,     0,            0,      0,             6*E*Iy/L**2,   0,      2*E*Iy/L,     0],
                    [0,      6*E*Iz/L**2,   0,             0,      0,            4*E*Iz/L,     0,      -6*E*Iz/L**2,  0,             0,      0,            2*E*Iz/L],
                    [-A*E/L, 0,             0,             0,      0,            0,            A*E/L,  0,             0,             0,      0,            0],
                    [0,      -12*E*Iz/L**3, 0,             0,      0,            -6*E*Iz/L**2, 0,      12*E*Iz/L**3,  0,             0,      0,            -6*E*Iz/L**2],
                    [0,      0,             -12*E*Iy/L**3, 0,      6*E*Iy/L**2,  0,            0,      0,             12*E*Iy/L**3,  0,      6*E*Iy/L**2,  0],
                    [0,      0,             0,             -G*J/L, 0,            0,            0,      0,             0,             G*J/L,  0,            0],
                    [0,      0,             -6*E*Iy/L**2,  0,      2*E*Iy/L,     0,            0,      0,             6*E*Iy/L**2,   0,      4*E*Iy/L,     0],
                    [0,      6*E*Iz/L**2,   0,             0,      0,            2*E*Iz/L,     0,      -6*E*Iz/L**2,  0,             0,      0,            4*E*Iz/L]])
        
        # Return the uncondensed local stiffness matrix
        return k

#%%
    def kg(self, P=0):
        """
        Returns the condensed (expanded) local geometric stiffness matrix for the member.

        Parameters
        ----------
        P : number, optional
            The axial force acting on the member (compression = +, tension = -)
        """

        # Get the properties needed to form the local geometric stiffness matrix
        Ip = self.section.Iy + self.section.Iz
        A = self.section.A
        L = self.L()
        
        # Create the uncondensed local geometric stiffness matrix
        kg = array([[0, 0,    0,     0,     0,         0,         0, 0,     0,    0,     0,         0],
                     [0, 6/5,  0,     0,     0,         L/10,      0, -6/5,  0,    0,     0,         L/10],
                     [0, 0,    6/5,   0,     -L/10,     0,         0, 0,     -6/5, 0,     -L/10,     0],
                     [0, 0,    0,     Ip/A,  0,         0,         0, 0,     0,    -Ip/A, 0,         0],
                     [0, 0,    -L/10, 0,     2*L**2/15, 0,         0, 0,     L/10, 0,     -L**2/30,  0],
                     [0, L/10, 0,     0,     0,         2*L**2/15, 0, -L/10, 0,    0,     0,         -L**2/30],
                     [0, 0,    0,     0,     0,         0,         0, 0,     0,    0,     0,         0],
                     [0, -6/5, 0,     0,     0,         -L/10,     0, 6/5,   0,    0,     0,         -L/10],
                     [0, 0,    -6/5,  0,     L/10,      0,         0, 0,     6/5,  0,     L/10,      0],
                     [0, 0,    0,     -Ip/A, 0,         0,         0, 0,     0,    Ip/A,  0,         0],
                     [0, 0,    -L/10, 0,     -L**2/30,  0,         0, 0,     L/10, 0,     2*L**2/15, 0],
                     [0, L/10, 0,     0,     0,         -L**2/30,  0, -L/10, 0,    0,     0,         2*L**2/15]])
        
        kg = kg*P/L

        # Partition the geometric stiffness matrix as 4 submatrices in
        # preparation for static condensation
        kg11, kg12, kg21, kg22 = self._partition(kg)
               
        # Calculate the condensed local geometric stiffness matrix
        # Note that a matrix of zeros cannot be inverted, so if P is 0 an error will occur
        if isclose(P, 0.0):
            kg_Condensed = zeros(kg11.shape)
        else:
            kg_Condensed = subtract(kg11, matmul(matmul(kg12, inv(kg22)), kg21))
        
        # Expand the condensed local geometric stiffness matrix
        i=0
        for DOF in self.Releases:
            
            if DOF == True:
                kg_Condensed = insert(kg_Condensed, i, 0, axis = 0)
                kg_Condensed = insert(kg_Condensed, i, 0, axis = 1)
                
            i += 1

        # Return the local geomtric stiffness matrix, with end releases applied
        return kg_Condensed
    
    def km(self, combo_name='Combo 1', push_combo='Push', step_num=1):
        """Returns the local plastic reduction matrix for the element.

        :param combo_name: The name of the load combination to get the plastic reduction matrix for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :param push_combo: The name of the load combination that defines the pushover load. Defaults to 'Push'.
        :type push_combo: str, optional
        :param step_num: The pushover load step to consider for calculating the plastic reduciton matrix. Default is 1.
        :type step_num: int, optional
        :return: The plastic reduction matrix for the element
        :rtype: array
        """
        
        # Get the elastic local stiffness matrix
        ke = self.k()

        # Get the member's axial force
        P = self._fxj - self._fxi

        # Get the geometric local stiffness matrix
        kg = self.kg(P)

        # Get the total elastic local stiffness matrix
        ke = add(ke, kg)

        # Get the gradient to the failure surface at at each end of the element
        if self.section is None:
            raise Exception('Nonlinear material analysis requires member sections to be defined. A section definition is missing for element ' + self.name + '.')
        else:
            Gi = self.section.G(self._fxi, self._myi, self._mzi)
            Gj = self.section.G(self._fxj, self._myj, self._mzj)

        # Expand the gradients for a 12 degree of freedom element
        zeros_array = zeros((6, 1))
        Gi = vstack((Gi, zeros_array))
        Gj = vstack((zeros_array, Gj))
        G = hstack((Gi, Gj))

        # Calculate the plastic reduction matrix for each end of the element
        # TODO: Note that `ke` below already accounts for P-Delta effects and any member end releases which should spill into `km`. I believe end releases will resolve themselves because of this. We'll see how this tests when we get to testing. If it causes problems when end releases are applied we may need to adjust our calculation of G when end releases are present.
        # Check that G is not a zero matrix, which indicates no plastic behavior
        if allclose(G, 0, atol=1e-14):
            return zeros((12, 12))
        else:
            return -ke @ G @ pinv(G.T @ ke @ G) @ G.T @ ke
    
    def lamb(self, model_Delta_D, combo_name='Combo 1', push_combo='Push', step_num=1):

        # Obtain the change in the member's end displacements from the calculated displacement change vector
        Delta_D = array([model_Delta_D[self.i_node.ID*6 + 0],
                        model_Delta_D[self.i_node.ID*6 + 1],
                        model_Delta_D[self.i_node.ID*6 + 2],
                        model_Delta_D[self.i_node.ID*6 + 3],
                        model_Delta_D[self.i_node.ID*6 + 4],
                        model_Delta_D[self.i_node.ID*6 + 5],
                        model_Delta_D[self.j_node.ID*6 + 0],
                        model_Delta_D[self.j_node.ID*6 + 1],
                        model_Delta_D[self.j_node.ID*6 + 2],
                        model_Delta_D[self.j_node.ID*6 + 3],
                        model_Delta_D[self.j_node.ID*6 + 4],
                        model_Delta_D[self.j_node.ID*6 + 5]]).reshape(12, 1)
        
        # Convert the gloabl changes in displacement to local coordinates
        Delta_d = self.T() @ Delta_D

        # Get the elastic local stiffness matrix
        ke = self.k()

        # Get the total end forces applied to the element
        f = self.f(combo_name, push_combo) - self.fer(combo_name) - self.fer(push_combo)*step_num

        # Get the gradient to the failure surface at at each end of the element
        if self.section is None:
            raise Exception('Nonlinear material analysis requires member sections to be defined. A section definition is missing for element ' + self.name + '.')
        else:
            if self.i_reversal == False:
                Gi = self.section.G(f[0], f[4], f[5])
            else:
                Gi = [[0], [0], [0]]
            
            if self.j_reversal == False:
                Gj = self.section.G(f[6], f[10], f[11])
            else:
                Gj = [[0], [0], [0]]
        
        # Expand the gradients for a 12 degree of freedom element
        zeros_array = zeros((6, 1))
        Gi = vstack((Gi, zeros_array))
        Gj = vstack((zeros_array, Gj))
        G = hstack((Gi, Gj))

        return inv(G.T @ ke @ G) @ G.T @ ke @ Delta_d

#%%
    def fer(self, combo_name='Combo 1'):
        """
        Returns the condensed (and expanded) local fixed end reaction vector for the member for the given load combination.

        Parameters
        ----------
        combo : LoadCombo
            The load combination to construct the fixed end reaction vector for.
        """
        
        # Get the lists of unreleased and released degree of freedom indices
        R1_indices, R2_indices = self._partition_D()

        # Partition the local stiffness matrix and local fixed end reaction vector
        k11, k12, k21, k22 = self._partition(self._k_unc())
        fer1, fer2 = self._partition(self._fer_unc(combo_name))
        
        # Calculate the condensed fixed end reaction vector
        ferCondensed = subtract(fer1, matmul(matmul(k12, inv(k22)), fer2))
        
        # Expand the condensed fixed end reaction vector
        i=0
        for DOF in self.Releases:
            
            if DOF == True:
                ferCondensed = insert(ferCondensed, i, 0, axis = 0)
                
            i += 1
        
        # Return the fixed end reaction vector        
        return ferCondensed
    
#%%
    def _fer_unc(self, combo_name='Combo 1'):
        """
        Returns the member's local fixed end reaction vector, ignoring the effects of end releases.
        Needed to apply the slope-deflection equation properly.
        """
        
        # Initialize the fixed end reaction vector
        fer = zeros((12,1))

        # Get the requested load combination
        combo = self.model.load_combos[combo_name]

        # Loop through each load case and factor in the load combination
        for case, factor in combo.factors.items():
        
            # Sum the fixed end reactions for the point loads & moments
            for ptLoad in self.PtLoads:

                # Check if the current point load corresponds to the current load case
                if ptLoad[3] == case:

                    if ptLoad[0] == 'Fx':
                        fer = add(fer, Pynite.FixedEndReactions.FER_AxialPtLoad(factor*ptLoad[1], ptLoad[2], self.L()))
                    elif ptLoad[0] == 'Fy':
                        fer = add(fer, Pynite.FixedEndReactions.FER_PtLoad(factor*ptLoad[1], ptLoad[2], self.L(), 'Fy'))
                    elif ptLoad[0] == 'Fz':
                        fer = add(fer, Pynite.FixedEndReactions.FER_PtLoad(factor*ptLoad[1], ptLoad[2], self.L(), 'Fz'))
                    elif ptLoad[0] == 'Mx':
                        fer = add(fer, Pynite.FixedEndReactions.FER_Torque(factor*ptLoad[1], ptLoad[2], self.L()))
                    elif ptLoad[0] == 'My':
                        fer = add(fer, Pynite.FixedEndReactions.FER_Moment(factor*ptLoad[1], ptLoad[2], self.L(), 'My'))
                    elif ptLoad[0] == 'Mz':     
                        fer = add(fer, Pynite.FixedEndReactions.FER_Moment(factor*ptLoad[1], ptLoad[2], self.L(), 'Mz'))
                    elif ptLoad[0] == 'FX' or ptLoad[0] == 'FY' or ptLoad[0] == 'FZ':
                        FX, FY, FZ = 0, 0, 0
                        if ptLoad[0] == 'FX': FX = 1
                        if ptLoad[0] == 'FY': FY = 1
                        if ptLoad[0] == 'FZ': FZ = 1
                        f = self.T()[:3, :][:, :3] @ array([FX*ptLoad[1], FY*ptLoad[1], FZ*ptLoad[1]])
                        fer = add(fer, Pynite.FixedEndReactions.FER_AxialPtLoad(factor*f[0], ptLoad[2], self.L()))
                        fer = add(fer, Pynite.FixedEndReactions.FER_PtLoad(factor*f[1], ptLoad[2], self.L(), 'Fy'))
                        fer = add(fer, Pynite.FixedEndReactions.FER_PtLoad(factor*f[2], ptLoad[2], self.L(), 'Fz'))
                    elif ptLoad[0] == 'MX' or ptLoad[0] == 'MY' or ptLoad[0] == 'MZ':
                        MX, MY, MZ = 0, 0, 0
                        if ptLoad[0] == 'MX': MX = 1
                        if ptLoad[0] == 'MY': MY = 1
                        if ptLoad[0] == 'MZ': MZ = 1
                        f = self.T()[:3, :][:, :3] @ array([MX*ptLoad[1], MY*ptLoad[1], MZ*ptLoad[1]])
                        fer = add(fer, Pynite.FixedEndReactions.FER_Torque(factor*f[0], ptLoad[2], self.L()))
                        fer = add(fer, Pynite.FixedEndReactions.FER_Moment(factor*f[1], ptLoad[2], self.L(), 'My'))
                        fer = add(fer, Pynite.FixedEndReactions.FER_Moment(factor*f[2], ptLoad[2], self.L(), 'Mz'))
                    else:
                        raise Exception('Invalid member point load direction specified.')
                
            # Sum the fixed end reactions for the distributed loads
            for distLoad in self.DistLoads:
                
                # Check if the current distributed load corresponds to the current load case
                if distLoad[5] == case:

                    if distLoad[0] == 'Fx':
                        fer = add(fer, Pynite.FixedEndReactions.FER_AxialLinLoad(factor*distLoad[1], factor*distLoad[2], distLoad[3], distLoad[4], self.L()))
                    elif distLoad[0] == 'Fy' or distLoad[0] == 'Fz':
                        fer = add(fer, Pynite.FixedEndReactions.FER_LinLoad(factor*distLoad[1], factor*distLoad[2], distLoad[3], distLoad[4], self.L(), distLoad[0]))
                    elif distLoad[0] == 'FX' or distLoad[0] == 'FY' or distLoad[0] == 'FZ':
                        FX, FY, FZ = 0, 0, 0
                        if distLoad[0] =='FX': FX = 1
                        if distLoad[0] =='FY': FY = 1
                        if distLoad[0] =='FZ': FZ = 1
                        w1 = self.T()[:3, :][:, :3] @ array([FX*distLoad[1], FY*distLoad[1], FZ*distLoad[1]])
                        w2 = self.T()[:3, :][:, :3] @ array([FX*distLoad[2], FY*distLoad[2], FZ*distLoad[2]])
                        fer = add(fer, Pynite.FixedEndReactions.FER_AxialLinLoad(factor*w1[0], factor*w2[0], distLoad[3], distLoad[4], self.L()))
                        fer = add(fer, Pynite.FixedEndReactions.FER_LinLoad(factor*w1[1], factor*w2[1], distLoad[3], distLoad[4], self.L(), 'Fy'))
                        fer = add(fer, Pynite.FixedEndReactions.FER_LinLoad(factor*w1[2], factor*w2[2], distLoad[3], distLoad[4], self.L(), 'Fz'))

        # Return the fixed end reaction vector, uncondensed
        return fer

#%%
    def _partition(self, unp_matrix):
        """
        Partitions a matrix into sub-matrices based on unreleased and released degree of freedom indices.
        """

        # Create auxiliary lists of released/unreleased DOFs
        R1_indices, R2_indices = self._partition_D()

        # Partition the matrix by slicing
        if unp_matrix.shape[1] == 1:
            m1 = unp_matrix[R1_indices, :]
            m2 = unp_matrix[R2_indices, :]
            return m1, m2
        else:
            m11 = unp_matrix[R1_indices, :][:, R1_indices]
            m12 = unp_matrix[R1_indices, :][:, R2_indices]
            m21 = unp_matrix[R2_indices, :][:, R1_indices]
            m22 = unp_matrix[R2_indices, :][:, R2_indices]
            return  m11, m12, m21, m22

#%%   
    def f(self, combo_name='Combo 1', push_combo='Push', step_num=1):
        """Returns the member's elastic local end force vector for the given load combination.

        :param combo_name: The load combination to get the local end for vector for. Defaults to 'Combo 1'.
        :type combo_name: str, optional
        :return: The member's local end force vector for the given load combination.
        :rtype: array
        """
        
        # Calculate and return the member's local end force vector
        if self.model.solution == 'P-Delta':
            # Back-calculate the axial force on the member from the axial strain
            P = (self.d(combo_name)[6, 0] - self.d(combo_name)[0, 0])*self.section.A*self.material.E/self.L()
            return add(matmul(add(self.k(), self.kg(P)), self.d(combo_name)), self.fer(combo_name))
        elif self.model.solution == 'Pushover':
            P = self._fxj - self._fxi
            return add(matmul(add(self.k(), self.kg(P), self.km(combo_name, push_combo, step_num)), self.d(combo_name)), self.fer(combo_name))
        else:
            return add(matmul(self.k(), self.d(combo_name)), self.fer(combo_name))

#%%
    def d(self, combo_name='Combo 1'):
        """
        Returns the member's local displacement vector.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to construct the displacement vector for (not the load combination itself).
        """
        
        # Calculate and return the local displacement vector
        return self.T() @ self.D(combo_name)
        
#%%  
    # Transformation matrix
    def T(self):
        """
        Returns the transformation matrix for the member.
        """

        x1 = self.i_node.X
        x2 = self.j_node.X
        y1 = self.i_node.Y
        y2 = self.j_node.Y
        z1 = self.i_node.Z
        z2 = self.j_node.Z
        L = self.L()
        
        # Calculate the direction cosines for the local x-axis
        x = [(x2-x1)/L, (y2-y1)/L, (z2-z1)/L]
            
        # Calculate the remaining direction cosines.
        # For now, the local z-axis will be kept parallel to the global XZ plane in all cases. It will be adjusted later if a rotation has been applied to the member.
        # Vertical members
        if isclose(x1, x2) and isclose(z1, z2):
            
            # For vertical members, keep the local y-axis in the XY plane to make 2D problems easier to solve in the XY plane
            if y2 > y1:
                y = [-1, 0, 0]
                z = [0, 0, 1]
            else:
                y = [1, 0, 0]
                z = [0, 0, 1]

        # Horizontal members
        elif isclose(y1, y2):
        
            # Find a vector in the direction of the local z-axis by taking the cross-product
            # of the local x-axis and the local y-axis. This vector will be perpendicular to
            # both the local x-axis and the local y-axis.
            y = [0, 1, 0]
            z = cross(x, y)

            # Divide the z-vector by its magnitude to produce a unit vector of direction cosines
            z = divide(z, (z[0]**2 + z[1]**2 + z[2]**2)**0.5)

        # Members neither vertical or horizontal
        else:

            # Find the projection of x on the global XZ plane
            proj = [x2-x1, 0, z2-z1]

            # Find a vector in the direction of the local z-axis by taking the cross-product
            # of the local x-axis and its projection on a plane parallel to the XZ plane. This
            # produces a vector perpendicular to both the local x-axis and its projection. This
            # vector will always be horizontal since it's parallel to the XZ plane. The order
            # in which the vectors are 'crossed' has been selected to ensure the y-axis always
            # has an upward component (i.e. the top of the beam is always on top).
            if y2 > y1:
                z = cross(proj, x)
            else:
                z = cross(x, proj)

            # Divide the z-vector by its magnitude to produce a unit vector of direction cosines
            z = divide(z, (z[0]**2 + z[1]**2 + z[2]**2)**0.5)
            
            # Find the direction cosines for the local y-axis
            y = cross(z, x)
            y = divide(y, (y[0]**2 + y[1]**2 + y[2]**2)**0.5)

        # Check if the member is rotated
        if self.rotation != 0.0:

            # Get the member rotation angle
            theta = radians(self.rotation)

            # Define the rotation matrix for rotation about the x-axis
            R = array([[1, 0, 0],
                    [0, cos(theta), -sin(theta)],
                    [0, sin(theta), cos(theta)]])

            # Rotate the y and z axes about the x axis
            y = R @ y
            z = R @ z

        # Create the direction cosines matrix
        dirCos = array([x, y, z])
      
        # Build the transformation matrix
        transMatrix = zeros((12, 12))
        transMatrix[0:3, 0:3] = dirCos
        transMatrix[3:6, 3:6] = dirCos
        transMatrix[6:9, 6:9] = dirCos
        transMatrix[9:12, 9:12] = dirCos
        
        return transMatrix

#%%
    # Member global stiffness matrix
    def K(self):
        """Returns the global elastic stiffness matrix for the member.

        :return: The global elastic stiffness matrix for the member.
        :rtype: array
        """
        
        # Calculate and return the stiffness matrix in global coordinates
        return matmul(matmul(inv(self.T()), self.k()), self.T())

    def Kg(self, P=0.0):
        """Returns the global geometric stiffness matrix for the member. Used for P-Delta analysis.

        :param P: Member axial load. Defaults to 0.
        :type P: float, optional
        :return: The global geometric stiffness matrix for the member.
        :rtype: array
        """
        
        # Calculate and return the geometric stiffness matrix in global coordinates
        return matmul(matmul(inv(self.T()), self.kg(P)), self.T())

    def Km(self, combo_name, push_combo, step_num):
        """Returns the global plastic reduction matrix for the member. Used to modify member behavior for plastic hinges at the ends.

        :param combo_name: The name of the load combination to get the plastic reduction matrix for.
        :type combo_name: string
        :param push_combo: The name of the load combination used to define the pushover load.
        :type push_combo: string
        :param step_num: The load step (1, 2, 3, ...etc) to use to determine the current load from the pushover combo.
        :type step_num: int
        :return: The global plastic reduction matrix for the member.
        :rtype: array
        """

        # Calculate and return the plastic reduction matrix in global coordinates
        return matmul(matmul(inv(self.T()), self.km(combo_name, push_combo, step_num)), self.T())
    
    def F(self, combo_name='Combo 1'):
        """
        Returns the member's global end force vector for the given load combination.
        """
        
        # Calculate and return the global force vector
        return matmul(inv(self.T()), self.f(combo_name))
    
    def FER(self, combo_name='Combo 1'):
        """
        Returns the global fixed end reaction vector

        Parameters
        ----------
        combo_name : string
            The name of the load combination to calculate the fixed end reaction vector for (not the load combination itself).
        """
        
        # Calculate and return the fixed end reaction vector
        return matmul(inv(self.T()), self.fer(combo_name))

#%%
    def D(self, combo_name='Combo 1'):
        """
        Returns the member's global displacement vector.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to construct the global
            displacement vector for (not the load combination itelf).
        """
        
        # Initialize the displacement vector
        D = zeros((12, 1))
        
        # TODO: I'm not sure this next block is the best way to handle inactive members - need to review
        # Read in the global displacements from the nodes
        # Apply axial displacements only if the member is active
        if self.active[combo_name] == True:
            D[0, 0] = self.i_node.DX[combo_name]
            D[6, 0] = self.j_node.DX[combo_name]

        # Apply the remaining displacements
        D[1, 0] = self.i_node.DY[combo_name]
        D[2, 0] = self.i_node.DZ[combo_name]
        D[3, 0] = self.i_node.RX[combo_name]
        D[4, 0] = self.i_node.RY[combo_name]
        D[5, 0] = self.i_node.RZ[combo_name]
        D[7, 0] = self.j_node.DY[combo_name]
        D[8, 0] = self.j_node.DZ[combo_name]
        D[9, 0] = self.j_node.RX[combo_name]
        D[10, 0] = self.j_node.RY[combo_name]
        D[11, 0] = self.j_node.RZ[combo_name]      

        # Return the global displacement vector
        return D

#%%
    def shear(self, Direction, x, combo_name='Combo 1'):
        """
        Returns the shear at a point along the member's length.
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the shear. Must be one of the following:
                'Fy' = Shear acting on the local y-axis.
                'Fz' = Shear acting on the local z-axis.
        x : number
            The location at which to find the shear.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]

            # Check which direction is of interest
            if Direction == 'Fy':
                
                # Check which segment 'x' falls on
                for segment in self.SegmentsZ:
                    if round(x, 10) >= round(segment.x1, 10) and round(x, 10) < round(segment.x2, 10):
                        return segment.Shear(x - segment.x1)
                    
                if isclose(x, self.L()):  
                    lastIndex = len(self.SegmentsZ) - 1
                    return self.SegmentsZ[lastIndex].Shear(x - self.SegmentsZ[lastIndex].x1)
                    
            elif Direction == 'Fz':
                
                for segment in self.SegmentsY:
                    
                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return segment.Shear(x - segment.x1)
                    
                if isclose(x, self.L()):
                    
                    lastIndex = len(self.SegmentsY) - 1
                    return self.SegmentsY[lastIndex].Shear(x - self.SegmentsY[lastIndex].x1)
        
        else:

            return 0
            
#%%
    def max_shear(self, Direction, combo_name='Combo 1'):
        """
        Returns the maximum shear in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the maximum shear. Must be one of the following:
                'Fy' = Shear acting on the local y-axis
                'Fz' = Shear acting on the local z-axis
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            if Direction == 'Fy':
                
                Vmax = self.SegmentsZ[0].Shear(0)

                for segment in self.SegmentsZ:
                    
                    if segment.max_shear() > Vmax:
                        
                        Vmax = segment.max_shear()
                        
            if Direction == 'Fz':
                
                Vmax = self.SegmentsY[0].Shear(0)

                for segment in self.SegmentsY:
                    
                    if segment.max_shear() > Vmax:
                        
                        Vmax = segment.max_shear()
            
            return Vmax
        
        else:

            return 0
    
#%%
    def min_shear(self, Direction, combo_name='Combo 1'):
        """
        Returns the minimum shear in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the minimum shear. Must be one of the following:
                'Fy' = Shear acting on the local y-axis
                'Fz' = Shear acting on the local z-axis
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]   
            
            if Direction == 'Fy':
                
                Vmin = self.SegmentsZ[0].Shear(0)

                for segment in self.SegmentsZ:
                    
                    if segment.min_shear() < Vmin:
                        
                        Vmin = segment.min_shear()
                        
            if Direction == 'Fz':
                
                Vmin = self.SegmentsY[0].Shear(0)

                for segment in self.SegmentsY:
                    
                    if segment.min_shear() < Vmin:
                        
                        Vmin = segment.min_shear()
            
            return Vmin
        
        else:

            return 0
    
#%%
    def plot_shear(self, Direction, combo_name='Combo 1', n_points=20):
        """
        Plots the shear diagram for the member
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the moment. Must be one of the following:
                'Fy' = Shear acting on the local y-axis.
                'Fz' = Shear acting on the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        n_points: int
            The number of points used to generate the plot
        """
        
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]
        
        # Import 'pyplot' if not already done
        if Member3D.__plt is None:
            from matplotlib import pyplot as plt
            Member3D.__plt = plt

        fig, ax = Member3D.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()
        
        x, V = self.shear_array(Direction, n_points, combo_name)

        Member3D.__plt.plot(x, V)
        Member3D.__plt.ylabel('Shear')
        Member3D.__plt.xlabel('Location')
        Member3D.__plt.title('Member ' + self.name + '\n' + combo_name)
        Member3D.__plt.show()    
        

    def shear_array(self, Direction: str, combo_name: str = 'Combo 1',
                     n_points: int = 0, x_array: ndarray | None = None) -> tuple[ndarray, ndarray]:
        """
        Returns the array of the shear in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction to plot the shear for. Must be one of the following:
                'Fy' = Shear acting on the local y-axis.
                'Fz' = Shear acting on the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        n_points: int
            The number of points in the array to generate over the full length of the member.        
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order
        """
        if Direction not in ('Fy', 'Fz'):
            raise ValueError(f"Direction must be either 'Fy' or 'Fz', not '{Direction}'")

        # Determine if a P-Delta analysis has been run
        if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
            # Include P-little-delta effects in the moment results
            P_delta = True
        else:
            # Do not include P-little delta effects in the moment results
            P_delta = False

        #Create x-array if needed
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array<0) or any(x_array>L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        #P-delta analysis is not vectorised yet, so do it element-wise
        if P_delta:
            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]

            y_arr = array([self.moment(Direction, x, combo_name) for x in x_array])
            return array([x_array, y_arr])
        
        #Extract load combination from model
        try:
            combo = self.model.load_combos[combo_name]
        except KeyError:
            raise NameError(f"No load combination named '{combo_name}' exists in the model")

        #Create function to convert load axes
        def convert_load_axis(force, load_direction):
            if load_direction == Direction:
                return force
            elif load_direction in ('FX', 'FY', 'FZ'):
                global_forces = zeros(3)
                if load_direction == 'FX': global_forces[0] = force
                if load_direction == 'FY': global_forces[1] = force
                if load_direction == 'FZ': global_forces[2] = force
                local_forces = self.T()[:3, :][:, :3] @ global_forces
                F = local_forces[1] if Direction == "Fy" else local_forces[2]
                return F
            else:
                return 0
                           
        #Create base results array
        f = self.f(combo_name) 
        V1 = f[1,0] if Direction == "Fy" else f[2,0]
        results = full(len(x_array), V1)

        #Extract point load_results
        for pt_load_Direction, P, x, case in self.PtLoads:
            #Check if load is part of the current load combination
            try:
                f = combo.factors[case]
            except KeyError:
                continue

            #Check the load is on the member
            if x < 0 or x > L:
                continue

            #Check the direction of the load is releveant to the desired result
            F = convert_load_axis(P * f, pt_load_Direction)
            if F == 0: continue

            #Partition x values into arrays either side of point load
            left_xvals = x_array[:searchsorted(x_array,x)]
            right_xvals = x_array[len(left_xvals):]

            #Create shear results and combine into results array
            vals_left = zeros(len(left_xvals))
            vals_right = full(len(right_xvals), F)

            results += hstack((vals_left,vals_right))

        #Extract distributed load_results
        for udl_load_Direction, w1, w2, x1, x2, case in self.DistLoads:
            #Check if load is part of the current load combination
            try:
                f = combo.factors[case]
            except KeyError:
                continue

            #Check the load is on the member
            if x2 < 0 or x1 > L:
                continue

            #Check the direction of the load is releveant to the desired result
            dir_factor = convert_load_axis(f, udl_load_Direction)
            if dir_factor == 0: continue


            x1 = max(x1, 0) #clip to member
            x2 = min(x2, L) #clip to member
            w1 = dir_factor * w1
            w2 = dir_factor * w2

            #Partition x values into arrays either side and internal to the distributed load
            left_index = searchsorted(x_array,x1)
            right_index = searchsorted(x_array,x2)
            left_xvals = x_array[:left_index]
            mid_xvals = x_array[left_index:right_index]
            right_xvals = x_array[right_index:]

            #Create shear results and combine into results array
            vals_left = zeros(len(left_xvals))

            w_at_x = w1 + (w2-w1) * (mid_xvals-x1) / (x2-x1)
            vals_mid = 0.5*(w1 + w_at_x) * (mid_xvals-x1)

            F = (x2-x1) * (w1+w2)/2
            vals_right = full(len(right_xvals), F)

            results += hstack((vals_left,vals_mid,vals_right))

        return array([x_array, results])
            

#%%
    def moment(self, Direction, x, combo_name='Combo 1'):
        """
        Returns the moment at a point along the member's length
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = moment about the local z-axis.
        x : number
            The location at which to find the moment.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            # Determine if a P-Delta analysis has been run
            if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
                # Include P-little-delta effects in the moment results
                P_delta = True
            else:
                # Do not include P-little delta effects in the moment results
                P_delta = False

            # Check which axis is of interest
            if Direction == 'My':
                
                # Check which segment 'x' falls on
                for segment in self.SegmentsY:

                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return segment.moment(x - segment.x1, P_delta)
                    
                if isclose(x, self.L()):
                    
                    return self.SegmentsY[-1].moment(x - self.SegmentsY[-1].x1, P_delta)
                    
            elif Direction == 'Mz':
                
                for segment in self.SegmentsZ:
                    
                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return segment.moment(x - segment.x1, P_delta)
                    
                if isclose(x, self.L()):
                    
                    return self.SegmentsZ[-1].moment(x - self.SegmentsZ[-1].x1, P_delta)
            
            else:
                raise ValueError(f"Direction must be 'My' or 'Mz'. {Direction} was given.")
    
        else:

            return 0
            
#%%
    def max_moment(self, Direction, combo_name='Combo 1'):
        """
        Returns the maximum moment in the member for the given direction.
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the maximum moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = Moment about the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """

        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Determine if a P-Delta analysis has been run
            if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
                # Include P-little-delta effects in the moment results
                P_delta = True
            else:
                # Do not include P-little delta effects in the moment results
                P_delta = False
            
            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            if Direction == 'Mz':
                
                Mmax = self.SegmentsZ[0].moment(0, P_delta)

                for segment in self.SegmentsZ:
                    
                    if segment.max_moment() > Mmax:
                        
                        Mmax = segment.max_moment()
                        
            if Direction == 'My':
                
                Mmax = self.SegmentsY[0].moment(0, P_delta)

                for segment in self.SegmentsY:
                    
                    if segment.max_moment() > Mmax:
                        
                        Mmax = segment.max_moment()
            
            return Mmax
        
        else:

            return 0

#%%
    def min_moment(self, Direction, combo_name='Combo 1'):
        """
        Returns the minimum moment in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the minimum moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = Moment about the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)   
                self._solved_combo = self.model.load_combos[combo_name]
            
            # Determine if a P-Delta analysis has been run
            if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
                # Include P-little-delta effects in the moment results
                P_delta = True
            else:
                # Do not include P-little delta effects in the moment results
                P_delta = False

            if Direction == 'Mz':
                
                Mmin = self.SegmentsZ[0].moment(0, P_delta)

                for segment in self.SegmentsZ:
                    
                    if segment.min_moment(P_delta) < Mmin: Mmin = segment.min_moment(P_delta)
                        
            if Direction == 'My':
                
                Mmin = self.SegmentsY[0].moment(0, P_delta)

                for segment in self.SegmentsY:
                    
                    if segment.min_moment(P_delta) < Mmin: Mmin = segment.min_moment(P_delta)
            
            return Mmin
        
        else:

            return 0

#%%
    def plot_moment(self, Direction, combo_name='Combo 1', n_points=20):
        """
        Plots the moment diagram for the member
        
        Parameters
        ----------
        Direction : string
            The direction in which to plot the moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = moment about the local z-axis.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        n_points: int
            The number of points used to generate the plot
        """
        
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]
                
        # Import 'pyplot' if not already done
        if Member3D.__plt is None:
            from matplotlib import pyplot as plt
            Member3D.__plt = plt
        
        fig, ax = Member3D.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()
        
        # Generate the moment diagram coordinates
        x, M = self.moment_array(Direction, n_points, combo_name)

        Member3D.__plt.plot(x, M)
        Member3D.__plt.ylabel('Moment')
        Member3D.__plt.xlabel('Location')
        Member3D.__plt.title('Member ' + self.name + '\n' + combo_name)
        Member3D.__plt.show()
    
    def moment_array(self, Direction: str, combo_name: str = 'Combo 1',
                     n_points: int = 0, x_array: ndarray | None = None) -> tuple[ndarray, ndarray]:
        """
        Returns the array of the moment in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the moment. Must be one of the following:
                'My' = Moment about the local y-axis.
                'Mz' = moment about the local z-axis.
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order.
        """
        if Direction not in ('My', 'Mz'):
            raise ValueError(f"Direction must be either 'My' or 'Mz', not '{Direction}'")

        # Determine if a P-Delta analysis has been run
        if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
            # Include P-little-delta effects in the moment results
            P_delta = True
        else:
            # Do not include P-little delta effects in the moment results
            P_delta = False

        #Create x-array if needed
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array<0) or any(x_array>L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        #P-delta analysis is not vectorised yet, so do it element-wise
        if P_delta:
            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]

            y_arr = array([self.moment(Direction, x, combo_name) for x in x_array])
            return array([x_array, y_arr])
        
        #Extract load combination from model
        try:
            combo = self.model.load_combos[combo_name]
        except KeyError:
            raise NameError(f"No load combination named '{combo_name}' exists in the model")

        #Create function to convert load axes
        local_force_dir = "Fy" if Direction == "Mz" else "Fz"
        def convert_load_axis(force, load_direction):
            if load_direction == local_force_dir or load_direction == Direction:
                return force
            
            elif load_direction in ('FX', 'FY', 'FZ'):
                global_forces = zeros(3)
                if load_direction == 'FX': global_forces[0] = force
                if load_direction == 'FY': global_forces[1] = force
                if load_direction == 'FZ': global_forces[2] = force
                local_forces = self.T()[:3, :][:, :3] @ global_forces
                F = local_forces[1] if local_force_dir == "Fy" else local_forces[2]
                return F
            
            elif load_direction in ('MX', 'MY', 'MZ'):
                global_forces = zeros(3)
                if load_direction == 'MX': global_forces[0] = force
                if load_direction == 'MY': global_forces[1] = force
                if load_direction == 'MZ': global_forces[2] = force
                local_forces = self.T()[:3, :][:, :3] @ global_forces
                F = local_forces[2] if Direction == "Mz" else local_forces[1]
                return F

            else:
                return 0
                           
        #Create base results array by interpolating moments between the two ends
        f = self.f(combo_name) 
        M1 = f[5, 0]  if Direction == "Mz" else -f[4, 0] 
        M2 = -f[11, 0]  if Direction == "Mz" else f[10, 0] 
        results = M1 + (M2-M1) * x_array/L

        #Extract point load_results
        for pt_load_Direction, P, x, case in self.PtLoads:
            #Check if load is part of the current load combination
            try:
                f = combo.factors[case]
            except KeyError:
                continue

            #Check the load is on the member
            if x < 0 or x > L:
                continue

            #Check the direction of the load is releveant to the desired result
            F = convert_load_axis(P * f, pt_load_Direction)
            if F == 0: continue

            #Partition x values into arrays either side of point load
            left_xvals = x_array[:searchsorted(x_array,x)]
            right_xvals = x_array[len(left_xvals):]

            #Create moment results and combine into results array
            if pt_load_Direction[0] == "M":
                R = F / L if Direction == "Mz" else -F / L
                vals_left = -R * left_xvals
                vals_right = R * (L-right_xvals)

            else:
                vals_left = F * (L-x) * left_xvals / L
                vals_right = F * x * (L-right_xvals) / L

            results += hstack((vals_left,vals_right))

        #Extract distributed load_results
        for udl_load_Direction, w1, w2, x1, x2, case in self.DistLoads:
            #Check if load is part of the current load combination
            try:
                f = combo.factors[case]
            except KeyError:
                continue

            #Check the load is on the member
            if x2 < 0 or x1 > L:
                continue

            #Check the direction of the load is releveant to the desired result
            dir_factor = convert_load_axis(f, udl_load_Direction)
            if dir_factor == 0: continue

            x1 = max(x1, 0) #clip to member
            x2 = min(x2, L) #clip to member
            w1 = dir_factor * w1
            w2 = dir_factor * w2

            #Check for trivial case, where the load is applied to the full member
            if x1 == 0 and abs(x2-L) < 1e-6:
                #Generate the uniform load component
                wmin = min(w1, w2)
                Ms = wmin * x_array * (L-x_array) / 2

                #If required, add in the triangular load component
                if w1 < w2:
                    dw = w2 - w1
                    W = dw * (x2-x1) / 2
                    Ms += W * x_array * (L**2-x_array**2)/(3*L**2)
                elif w1 > w2:
                    dw = w1 - w2
                    W = dw * (x2-x1) / 2
                    Ms += W * (L-x_array) * (L**2-(L-x_array)**2)/(3*L**2)

                results += Ms
                continue

            #Partition x values into arrays either side and internal to the distributed load
            left_index = searchsorted(x_array,x1)
            right_index = searchsorted(x_array,x2)
            left_xvals = x_array[:left_index]
            mid_xvals = x_array[left_index:right_index]
            right_xvals = x_array[right_index:]

            #Create moment results and combine into results array
            if w1 == w2:
                w = w1
                vals_left = -w*left_xvals*x1 + w*left_xvals*x2 + \
                    w*left_xvals*x1**2/(2*L) - w*left_xvals*x2**2/(2*L)

                vals_mid = -w*mid_xvals**2 + w*mid_xvals*x2 + w*mid_xvals**3/(2*L) - \
                    w*mid_xvals*x2**2/(2*L) + mid_xvals**2*(L*w - w*mid_xvals)/(2*L) - \
                    x1**2*(L*w - w*mid_xvals)/(2*L)

                vals_right = -x1**2*(L*w - w*right_xvals)/(2*L) + \
                    x2**2*(L*w - w*right_xvals)/(2*L)
                
            else:
                #Equations are more complex if load varies over length of beam
                vals_left = -x1**3*(-left_xvals*w1 + left_xvals*w2)/(3*L*x1 - 3*L*x2) - \
                    x1**2*(L*left_xvals*w1 - L*left_xvals*w2 + left_xvals*w1*x2 - \
                    left_xvals*w2*x1)/(2*L*x1 - 2*L*x2) - x1*(-left_xvals*w1*x2 + \
                    left_xvals*w2*x1)/(x1 - x2) + x2**3*(-left_xvals*w1 + left_xvals*w2)/(3*L*x1 - \
                    3*L*x2) + x2**2*(L*left_xvals*w1 - L*left_xvals*w2 + left_xvals*w1*x2 - \
                    left_xvals*w2*x1)/(2*L*x1 - 2*L*x2) + x2*(-left_xvals*w1*x2 + left_xvals*w2*x1)/(x1 - x2)
                
                vals_mid = -mid_xvals**3*(-mid_xvals*w1 + mid_xvals*w2)/(3*L*x1 - 3*L*x2) + \
                    mid_xvals**3*(L*w1 - L*w2 - mid_xvals*w1 + mid_xvals*w2)/(3*L*x1 - 3*L*x2) - \
                    mid_xvals**2*(L*mid_xvals*w1 - L*mid_xvals*w2 + mid_xvals*w1*x2 - \
                    mid_xvals*w2*x1)/(2*L*x1 - 2*L*x2) + mid_xvals**2*(-L*w1*x2 + L*w2*x1 + \
                    mid_xvals*w1*x2 - mid_xvals*w2*x1)/(2*L*x1 - 2*L*x2) - \
                    mid_xvals*(-mid_xvals*w1*x2 + mid_xvals*w2*x1)/(x1 - x2) - x1**3*(L*w1 -
                    L*w2 - mid_xvals*w1 + mid_xvals*w2)/(3*L*x1 - 3*L*x2) - x1**2*(-L*w1*x2 + \
                    L*w2*x1 + mid_xvals*w1*x2 - mid_xvals*w2*x1)/(2*L*x1 - 2*L*x2) + \
                    x2**3*(-mid_xvals*w1 + mid_xvals*w2)/(3*L*x1 - 3*L*x2) + \
                    x2**2*(L*mid_xvals*w1 - L*mid_xvals*w2 + mid_xvals*w1*x2 - \
                    mid_xvals*w2*x1)/(2*L*x1 - 2*L*x2) + x2*(-mid_xvals*w1*x2 + \
                    mid_xvals*w2*x1)/(x1 - x2)
                
                vals_right = -x1**3*(L*w1 - L*w2 - right_xvals*w1 + right_xvals*w2)/(3*L*x1 - \
                    3*L*x2) - x1**2*(-L*w1*x2 + L*w2*x1 + right_xvals*w1*x2 - \
                    right_xvals*w2*x1)/(2*L*x1 - 2*L*x2) + x2**3*(L*w1 - L*w2 - right_xvals*w1 + \
                    right_xvals*w2)/(3*L*x1 - 3*L*x2) + x2**2*(-L*w1*x2 + L*w2*x1 + \
                    right_xvals*w1*x2 - right_xvals*w2*x1)/(2*L*x1 - 2*L*x2)

            results += hstack((vals_left,vals_mid,vals_right))

        return array([x_array, results])
       
#%%
    def torque(self, x, combo_name='Combo 1'):
        """
        Returns the torsional moment at a point along the member's length
        
        Parameters
        ----------
        x : number
            The location at which to find the torque
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
                
            # Check which segment 'x' falls on
            for segment in self.SegmentsX:
                if round(x, 10) >= round(segment.x1, 10) and round(x, 10) < round(segment.x2, 10):
                    return segment.Torsion()
                    
                if isclose(x, self.L()):  
                    lastIndex = len(self.SegmentsX) - 1
                    return self.SegmentsX[lastIndex].Torsion()
        
        else:

            return 0

#%%
    def max_torque(self, combo_name='Combo 1'):
        """
        Returns the maximum torsional moment in the member.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]       
            
            Tmax = self.SegmentsX[0].Torsion()   
            
            for segment in self.SegmentsX:

                if segment.MaxTorsion() > Tmax:
                        
                    Tmax = segment.MaxTorsion()
            
            return Tmax
        
        else:

            return 0
    
#%%
    def min_torque(self, combo_name='Combo 1'):
        """
        Returns the minimum torsional moment in the member.

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            Tmin = self.SegmentsX[0].Torsion()
                
            for segment in self.SegmentsX:
                    
                if segment.MinTorsion() < Tmin:
                        
                    Tmin = segment.MinTorsion()
            
            return Tmin
        
        else:

            return 0

#%%
    def plot_torque(self, combo_name='Combo 1', n_points=20):
        """
        Plots the torque diagram for the member.
        
        Paramters
        ---------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        n_points: int
            The number of points used to generate the plot
        """
        
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]
        
        # Import 'pyplot' if not already done
        if Member3D.__plt is None:
            from matplotlib import pyplot as plt
            Member3D.__plt = plt

        fig, ax = Member3D.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()
        
        x, T = self.torque_array(n_points, combo_name)

        Member3D.__plt.plot(x, T)
        Member3D.__plt.ylabel('Torsional Moment (Warping Torsion Not Included)') # Torsion results are for pure torsion. Torsional warping has not been considered
        Member3D.__plt.xlabel('Location')
        Member3D.__plt.title('Member ' + self.name + '\n' + combo_name)
        Member3D.__plt.show()

    def torque_array(self, n_points, combo_name='Combo 1', x_array = None):
        """
        Returns the array of the torque in the member
        
        Parameters
        ----------
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order.
        """
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]

        L = self.L()

        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array<0) or any(x_array>L):
                raise ValueError(f"All x values must be in the range 0 to {L}")
            
        return self._extract_vector_results(self.SegmentsX, x_array, 'torque')

        
    def axial(self, x, combo_name='Combo 1'):
        """
        Returns the axial force at a point along the member's length.
        
        Parameters
        ----------
        x : number
            The location at which to find the axial force.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
                
            # Check which segment 'x' falls on
            for segment in self.SegmentsZ:
                if round(x, 10) >= round(segment.x1, 10) and round(x, 10) < round(segment.x2, 10):
                    return segment.axial(x - segment.x1)
                    
                if isclose(x, self.L()):  
                    lastIndex = len(self.SegmentsZ) - 1
                    return self.SegmentsZ[lastIndex].axial(x - self.SegmentsZ[lastIndex].x1)
                
        else:

            return 0

    def max_axial(self, combo_name='Combo 1'):
        """
        Returns the maximum axial force in the member

        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            Pmax = self.SegmentsZ[0].axial(0)   
            
            for segment in self.SegmentsZ:

                if segment.max_axial() > Pmax:
                        
                    Pmax = segment.max_axial()
            
            return Pmax
        
        else:

            return 0
    
    def min_axial(self, combo_name='Combo 1'):
        """
        Returns the minimum axial force in the member.
        
        Paramters
        ---------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            Pmin = self.SegmentsZ[0].axial(0)
                
            for segment in self.SegmentsZ:
                    
                if segment.min_axial() < Pmin:
                        
                    Pmin = segment.min_axial()
            
            return Pmin
        
        else:

            return 0
    
    def plot_axial(self, combo_name='Combo 1', n_points=20):
        """
        Plots the axial force diagram for the member.
        
        Parameters
        ----------
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        n_points: int
            The number of points used to generate the plot
        """

        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]
        
        # Import 'pyplot' if not already done
        if Member3D.__plt is None:
            from matplotlib import pyplot as plt
            Member3D.__plt = plt

        fig, ax = Member3D.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()
        
        x, P = self.axial_array(n_points, combo_name)

        Member3D.__plt.plot(x, P)
        Member3D.__plt.ylabel('Axial Force')
        Member3D.__plt.xlabel('Location')
        Member3D.__plt.title('Member ' + self.name + '\n' + combo_name)
        Member3D.__plt.show()    

    def axial_array(self, n_points, combo_name='Combo 1', x_array=None):
        """
        Returns the array of the axial force in the member for the given direction
        
        Parameters
        ----------
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order.
        """

        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]

        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array<0) or any(x_array>L):
                raise ValueError(f"All x values must be in the range 0 to {L}")
            
        return self._extract_vector_results(self.SegmentsZ, x_array, 'axial')

                        
    def deflection(self, Direction, x, combo_name='Combo 1'):
        """
        Returns the deflection at a point along the member's length.
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the deflection. Must be one of the following:
                'dx' = Deflection in the local x-axis.
                'dy' = Deflection in the local y-axis.
                'dz' = Deflection in the local z-axis.
        x : number
            The location at which to find the deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
                P_delta = True
            else:
                P_delta = False

            # Check which axis is of interest
            if Direction == 'dx':
                
                # Check which segment 'x' falls on
                for segment in self.SegmentsZ:
                    
                    if round(x, 10) >= round(segment.x1, 10) and round(x, 10) < round(segment.x2, 10):
                        return segment.AxialDeflection(x - segment.x1)
                    
                if isclose(x, self.L()):
                    
                    lastIndex = len(self.SegmentsZ) - 1
                    return self.SegmentsZ[lastIndex].AxialDeflection(x - self.SegmentsZ[lastIndex].x1)

            elif Direction == 'dy':
                
                # Check which segment 'x' falls on
                for segment in self.SegmentsZ:
                    
                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return segment.deflection(x - segment.x1, P_delta)
                    
                if isclose(x, self.L()):
                    
                    lastIndex = len(self.SegmentsZ) - 1
                    return self.SegmentsZ[lastIndex].deflection(x - self.SegmentsZ[lastIndex].x1, P_delta)
                    
            elif Direction == 'dz':
                
                for segment in self.SegmentsY:
                    
                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return segment.deflection(x - segment.x1)
                    
                if isclose(x, self.L()):
                    
                    lastIndex = len(self.SegmentsY) - 1
                    return self.SegmentsY[lastIndex].deflection(x - self.SegmentsY[lastIndex].x1) 
        
        else:

            return 0

    def max_deflection(self, Direction, combo_name='Combo 1'):
        """
        Returns the maximum deflection in the member.
        
        Parameters
        ----------
        Direction : {'dy', 'dz'}
            The direction in which to find the maximum deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            # Initialize the maximum deflection
            dmax = self.deflection(Direction, 0, combo_name)
            
            # Check the deflection at 100 locations along the member and find the largest value
            for i in range(100):
                d = self.deflection(Direction, self.L()*i/99, combo_name)
                if d > dmax:
                    dmax = d
            
            # Return the largest value
            return dmax
        
        else:

            return 0
    
    def min_deflection(self, Direction, combo_name='Combo 1'):
        """
        Returns the minimum deflection in the member.
        
        Parameters
        ----------
        Direction : {'dy', 'dz'}
            The direction in which to find the minimum deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        """
        
        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            # Initialize the minimum deflection
            dmin = self.deflection(Direction, 0, combo_name)
            
            # Check the deflection at 100 locations along the member and find the smallest value
            for i in range(100):
                d = self.deflection(Direction, self.L()*i/99, combo_name)
                if d < dmin:
                    dmin = d
            
            # Return the smallest value
            return dmin
        
        else:

            return 0
              
    def plot_deflection(self, Direction, combo_name='Combo 1', n_points=20):
        """
        Plots the deflection diagram for the member
        
        Parameters
        ----------
        Direction : {'dy', 'dz'}
            The direction in which to plot the deflection.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        n_points: int
            The number of points used to generate the plot
        """
        
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]
                
        # Import 'pyplot' if not already done
        if Member3D.__plt is None:
            from matplotlib import pyplot as plt
            Member3D.__plt = plt
        
        fig, ax = Member3D.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()
        
        x, d = self.deflection_array(Direction, n_points, combo_name)

        Member3D.__plt.plot(x, d)
        Member3D.__plt.ylabel('Deflection')
        Member3D.__plt.xlabel('Location')
        Member3D.__plt.title('Member ' + self.name + '\n' + combo_name)
        Member3D.__plt.show()

    def deflection_array(self, Direction: str, combo_name: str = 'Combo 1',
                     n_points: int = 0, x_array: ndarray | None = None) -> tuple[ndarray, ndarray]:
        """
        Returns the array of the deflection in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the deflection. Must be one of the following:
                'dx' = Deflection in the local x-axis.
                'dy' = Deflection in the local y-axis.
                'dz' = Deflection in the local z-axis.
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order.
        """
        if Direction not in ('dx', 'dy', 'dz'):
            raise ValueError(f"Direction must be either 'dx', 'dy' or 'dz', not '{Direction}'")

        # Determine if a P-Delta analysis has been run
        if self.model.solution == 'P-Delta' or self.model.solution == 'Pushover':
            # Include P-little-delta effects in the moment results
            P_delta = True
        else:
            # Do not include P-little delta effects in the moment results
            P_delta = False

        #Create x-array if needed
        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array<0) or any(x_array>L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        #Axial deflections and P-delta analysis is not vectorised yet, so do it element-wise
        if Direction == 'dx' or P_delta:
            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]

            y_arr = array([self.deflection(Direction, x, combo_name) for x in x_array])
            return array([x_array, y_arr])
        
        #Extract load combination from model
        try:
            combo = self.model.load_combos[combo_name]
        except KeyError:
            raise NameError(f"No load combination named '{combo_name}' exists in the model")

        #Create function to convert load axes
        if Direction == 'dy':
            local_force_dir = "Fy"
            local_mom_dir = 'Mz'
        else:
            local_force_dir = "Fz"
            local_mom_dir = 'My'

        def convert_load_axis(force, load_direction):
            if load_direction == local_force_dir or load_direction == local_mom_dir:
                return force
            
            elif load_direction in ('FX', 'FY', 'FZ'):
                global_forces = zeros(3)
                if load_direction == 'FX': global_forces[0] = force
                if load_direction == 'FY': global_forces[1] = force
                if load_direction == 'FZ': global_forces[2] = force
                local_forces = self.T()[:3, :][:, :3] @ global_forces
                F = local_forces[1] if local_force_dir == "Fy" else local_forces[2]
                return F
            
            elif load_direction in ('MX', 'MY', 'MZ'):
                global_forces = zeros(3)
                if load_direction == 'MX': global_forces[0] = force
                if load_direction == 'MY': global_forces[1] = force
                if load_direction == 'MZ': global_forces[2] = force
                local_forces = self.T()[:3, :][:, :3] @ global_forces
                F = local_forces[1] if local_mom_dir == "My" else local_forces[2]
                return F

            else:
                return 0
                           
        #Create base results array by interpolating deflections between the two ends
        f = self.f(combo_name) 
        d = self.d(combo_name)
        if Direction == "dy":
            M1 = f[5, 0]  
            M2 = -f[11, 0]
            d1 = d[1,0]
            d2 = d[7,0]
            EI = self.material.E * self.section.Iz
        else:
            M1 = -f[4, 0] 
            M2 = f[10, 0] 
            d1 = d[2,0]
            d2 = d[8,0]
            EI = self.material.E * self.section.Iy
        
        results = d1 + (d2-d1) * x_array/L

        #Add in deflections due to fixed end moments
        results += M1 * L * (L-x_array) * (1-(L-x_array)**2/L**2)/(6*EI)
        results += M2 * L * x_array * (1-x_array**2/L**2)/(6*EI)

        #Extract point load_results
        for pt_load_Direction, P, x, case in self.PtLoads:
            #Check if load is part of the current load combination
            try:
                f = combo.factors[case]
            except KeyError:
                continue

            #Check the load is on the member
            if x < 0 or x > L:
                continue

            #Check the direction of the load is releveant to the desired result
            F = convert_load_axis(P * f, pt_load_Direction)
            if F == 0: continue

            #Partition x values into arrays either side of point load
            left_xvals = x_array[:searchsorted(x_array,x)]
            right_xvals = x_array[len(left_xvals):]

            #Create moment results and combine into results array
            if pt_load_Direction[0] == "M":
                sgn = 1 if Direction == 'dz' else -1
                vals_left = sgn*F * ((6*x - 3*x**2/L - 2*L)*left_xvals - left_xvals**3/L)/(6*EI)
                vals_right = sgn*F * (3*(x**2+right_xvals**2) - right_xvals**3/L - (2*L + 3* x**2/L)*right_xvals)/(6*EI)

            else:
                vals_left = F * (L-x) * left_xvals * (L**2 - (L-x)**2 - left_xvals**2) / (6*EI*L)
                vals_right = F * x * (L-right_xvals) * (L**2 - x**2 - (L-right_xvals)**2) / (6*EI*L)

            results += hstack((vals_left,vals_right))

        #Extract distributed load_results
        for udl_load_Direction, w1, w2, x1, x2, case in self.DistLoads:
            #Check if load is part of the current load combination
            try:
                f = combo.factors[case]
            except KeyError:
                continue

            #Check the load is on the member
            if x2 < 0 or x1 > L:
                continue

            #Check the direction of the load is releveant to the desired result
            dir_factor = convert_load_axis(f, udl_load_Direction)
            if dir_factor == 0: continue

            x1 = max(x1, 0) #clip to member
            x2 = min(x2, L) #clip to member
            w1 = dir_factor * w1
            w2 = dir_factor * w2

            #Check for trivial case, where the load is applied to the full member
            if x1 == 0 and abs(x2-L) < 1e-6:
                #Generate the uniform load component
                wmin = min(w1, w2)
                ds = wmin*x_array*(L**3-2*L*x_array**2+x_array**3)/(24*EI)

                #If required, add in the triangular load component
                if w1 < w2:
                    dw = w2 - w1
                    W = dw * (x2-x1) / 2
                    ds += W * x_array * (3*x_array**4-10*L**2*x_array**2+7*L**4)/(180*EI*L**2)

                elif w1 > w2:
                    dw = w1 - w2
                    W = dw * (x2-x1) / 2
                    ds += W * (L-x_array) * (3*(L-x_array)**4-10*L**2*(L-x_array)**2+7*L**4)/(180*EI*L**2)

                results += ds
                continue

            #Partition x values into arrays either side and internal to the distributed load
            left_index = searchsorted(x_array,x1)
            right_index = searchsorted(x_array,x2)
            left_xvals = x_array[:left_index]
            mid_xvals = x_array[left_index:right_index]
            right_xvals = x_array[right_index:]

            #Create moment results and combine into results array
            if w1 == w2:
                w = w1
                vals_left = left_xvals**3*w*x1/(6*EI) - left_xvals**3*w*x2/(6*EI) + left_xvals*w*x1**3/(6*EI) - \
                    left_xvals*w*x2**3/(6*EI) - left_xvals*w*x1**4/(24*EI*L) + left_xvals*w*x2**4/(24*EI*L) - \
                    x1**2*(2*L**2*left_xvals*w + left_xvals**3*w)/(12*EI*L) + x2**2*(2*L**2*left_xvals*w + \
                    left_xvals**3*w)/(12*EI*L)

                vals_mid = mid_xvals**4*w/(3*EI) - mid_xvals**3*w*x2/(6*EI) - mid_xvals*w*x2**3/(6*EI) - \
                    mid_xvals**5*w/(24*EI*L) + mid_xvals**4*(-L*w + mid_xvals*w)/(24*EI*L) - \
                    mid_xvals**2*(2*L**2*mid_xvals*w + mid_xvals**3*w)/(12*EI*L) + mid_xvals**2*(2*L**2*mid_xvals*w - \
                    3*L*mid_xvals**2*w + mid_xvals**3*w)/(12*EI*L) + mid_xvals*w*x2**4/(24*EI*L) - \
                    x1**4*(-L*w + mid_xvals*w)/(24*EI*L) - x1**2*(2*L**2*mid_xvals*w - \
                    3*L*mid_xvals**2*w + mid_xvals**3*w)/(12*EI*L) + x2**2*(2*L**2*mid_xvals*w + mid_xvals**3*w)/(12*EI*L)

                vals_right = -x1**4*(-L*w + right_xvals*w)/(24*EI*L) - x1**2*(2*L**2*right_xvals*w - 3*L*right_xvals**2*w + \
                    right_xvals**3*w)/(12*EI*L) + x2**4*(-L*w + right_xvals*w)/(24*EI*L) + x2**2*(2*L**2*right_xvals*w - \
                    3*L*right_xvals**2*w + right_xvals**3*w)/(12*EI*L)
                
            else:
                #Equations are extremely complex if load varies over length of beam
                vals_left = -x1**5*(left_xvals*w1 - left_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) - \
                    x1**4*(-3*L*left_xvals*w1 + 3*L*left_xvals*w2 - left_xvals*w1*x2 + \
                    left_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) - x1**3*(2*L**2*left_xvals*w1 - \
                    2*L**2*left_xvals*w2 + 3*L*left_xvals*w1*x2 - 3*L*left_xvals*w2*x1 + left_xvals**3*w1 - \
                    left_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) - x1**2*(-2*L**2*left_xvals*w1*x2 + \
                    2*L**2*left_xvals*w2*x1 - L*left_xvals**3*w1 + L*left_xvals**3*w2 - \
                    left_xvals**3*w1*x2 + left_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2) - \
                    x1*(left_xvals**3*w1*x2 - left_xvals**3*w2*x1)/(6*EI*x1 - 6*EI*x2) + x2**5*(left_xvals*w1 - \
                    left_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) + x2**4*(-3*L*left_xvals*w1 + 3*L*left_xvals*w2 - \
                    left_xvals*w1*x2 + left_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) + x2**3*(2*L**2*left_xvals*w1 - \
                    2*L**2*left_xvals*w2 + 3*L*left_xvals*w1*x2 - 3*L*left_xvals*w2*x1 + left_xvals**3*w1 - \
                    left_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) + x2**2*(-2*L**2*left_xvals*w1*x2 + \
                    2*L**2*left_xvals*w2*x1 - L*left_xvals**3*w1 + L*left_xvals**3*w2 - left_xvals**3*w1*x2 + \
                    left_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2) + x2*(left_xvals**3*w1*x2 - \
                    left_xvals**3*w2*x1)/(6*EI*x1 - 6*EI*x2)
                
                vals_mid = -mid_xvals**5*(mid_xvals*w1 - mid_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) + \
                    mid_xvals**5*(-L*w1 + L*w2 + mid_xvals*w1 - mid_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) - \
                    mid_xvals**4*(-3*L*mid_xvals*w1 + 3*L*mid_xvals*w2 - mid_xvals*w1*x2 + \
                    mid_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) + mid_xvals**4*(L*w1*x2 - L*w2*x1 - \
                    mid_xvals*w1*x2 + mid_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) + mid_xvals**3*(2*L**2*mid_xvals*w1 - \
                    2*L**2*mid_xvals*w2 - 3*L*mid_xvals**2*w1 + 3*L*mid_xvals**2*w2 + mid_xvals**3*w1 - \
                    mid_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) - mid_xvals**3*(2*L**2*mid_xvals*w1 - \
                    2*L**2*mid_xvals*w2 + 3*L*mid_xvals*w1*x2 - 3*L*mid_xvals*w2*x1 + mid_xvals**3*w1 - \
                    mid_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) - mid_xvals**2*(-2*L**2*mid_xvals*w1*x2 + \
                    2*L**2*mid_xvals*w2*x1 - L*mid_xvals**3*w1 + L*mid_xvals**3*w2 - mid_xvals**3*w1*x2 + \
                    mid_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2) + mid_xvals**2*(-2*L**2*mid_xvals*w1*x2 + \
                    2*L**2*mid_xvals*w2*x1 + 3*L*mid_xvals**2*w1*x2 - 3*L*mid_xvals**2*w2*x1 - \
                    mid_xvals**3*w1*x2 + mid_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2) - \
                    mid_xvals*(mid_xvals**3*w1*x2 - mid_xvals**3*w2*x1)/(6*EI*x1 - 6*EI*x2) - \
                    x1**5*(-L*w1 + L*w2 + mid_xvals*w1 - mid_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) - \
                    x1**4*(L*w1*x2 - L*w2*x1 - mid_xvals*w1*x2 + mid_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) - \
                    x1**3*(2*L**2*mid_xvals*w1 - 2*L**2*mid_xvals*w2 - 3*L*mid_xvals**2*w1 + \
                    3*L*mid_xvals**2*w2 + mid_xvals**3*w1 - mid_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) - \
                    x1**2*(-2*L**2*mid_xvals*w1*x2 + 2*L**2*mid_xvals*w2*x1 + 3*L*mid_xvals**2*w1*x2 - \
                    3*L*mid_xvals**2*w2*x1 - mid_xvals**3*w1*x2 + mid_xvals**3*w2*x1)/(12*EI*L*x1 - \
                    12*EI*L*x2) + x2**5*(mid_xvals*w1 - mid_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) + \
                    x2**4*(-3*L*mid_xvals*w1 + 3*L*mid_xvals*w2 - mid_xvals*w1*x2 + \
                    mid_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) + x2**3*(2*L**2*mid_xvals*w1 - \
                    2*L**2*mid_xvals*w2 + 3*L*mid_xvals*w1*x2 - 3*L*mid_xvals*w2*x1 + mid_xvals**3*w1 - \
                    mid_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) + x2**2*(-2*L**2*mid_xvals*w1*x2 + \
                    2*L**2*mid_xvals*w2*x1 - L*mid_xvals**3*w1 + L*mid_xvals**3*w2 - mid_xvals**3*w1*x2 + \
                    mid_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2) + x2*(mid_xvals**3*w1*x2 - \
                    mid_xvals**3*w2*x1)/(6*EI*x1 - 6*EI*x2)
                
                vals_right = -x1**5*(-L*w1 + L*w2 + right_xvals*w1 - right_xvals*w2)/(30*EI*L*x1 - \
                    30*EI*L*x2) - x1**4*(L*w1*x2 - L*w2*x1 - right_xvals*w1*x2 + \
                    right_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) - x1**3*(2*L**2*right_xvals*w1 - \
                    2*L**2*right_xvals*w2 - 3*L*right_xvals**2*w1 + 3*L*right_xvals**2*w2 + \
                    right_xvals**3*w1 - right_xvals**3*w2)/(18*EI*L*x1 - 18*EI*L*x2) - \
                    x1**2*(-2*L**2*right_xvals*w1*x2 + 2*L**2*right_xvals*w2*x1 + \
                    3*L*right_xvals**2*w1*x2 - 3*L*right_xvals**2*w2*x1 - right_xvals**3*w1*x2 + \
                    right_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2) + x2**5*(-L*w1 + L*w2 + \
                    right_xvals*w1 - right_xvals*w2)/(30*EI*L*x1 - 30*EI*L*x2) + x2**4*(L*w1*x2 - \
                    L*w2*x1 - right_xvals*w1*x2 + right_xvals*w2*x1)/(24*EI*L*x1 - 24*EI*L*x2) + \
                    x2**3*(2*L**2*right_xvals*w1 - 2*L**2*right_xvals*w2 - 3*L*right_xvals**2*w1 + \
                    3*L*right_xvals**2*w2 + right_xvals**3*w1 - right_xvals**3*w2)/(18*EI*L*x1 - \
                    18*EI*L*x2) + x2**2*(-2*L**2*right_xvals*w1*x2 + 2*L**2*right_xvals*w2*x1 + \
                    3*L*right_xvals**2*w1*x2 - 3*L*right_xvals**2*w2*x1 - right_xvals**3*w1*x2 + \
                    right_xvals**3*w2*x1)/(12*EI*L*x1 - 12*EI*L*x2)

            results += hstack((vals_left,vals_mid,vals_right))

        return array([x_array, results])

    def rel_deflection(self, Direction, x, combo_name='Combo 1'):
        """
        Returns the relative deflection at a point along the member's length
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the relative deflection. Must be one of the following:
                'dy' = Deflection in the local y-axis
                'dz' = Deflection in the local z-axis
        x : number
            The location at which to find the relative deflection
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """

        # Only calculate results if the member is currently active
        if self.active[combo_name]:

            # Segment the member if necessary
            if self._solved_combo is None or combo_name != self._solved_combo.name:
                self._segment_member(combo_name)
                self._solved_combo = self.model.load_combos[combo_name]
            
            d = self.d(combo_name)
            dyi = d[1,0]
            dyj = d[7,0]
            dzi = d[2,0]
            dzj = d[8,0]
            L = self.L()
        
            # Check which axis is of interest
            if Direction == 'dy':
                
                # Check which segment 'x' falls on
                for segment in self.SegmentsZ:
                    
                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return (segment.deflection(x - segment.x1)) - (dyi + (dyj-dyi)/L*x)
                    
                if isclose(x, self.L()):
                    
                    lastIndex = len(self.SegmentsZ) - 1
                    return (self.SegmentsZ[lastIndex].deflection(x - self.SegmentsZ[lastIndex].x1))-dyj
                    
            elif Direction == 'dz':
                
                for segment in self.SegmentsY:
                    
                    if round(x,10) >= round(segment.x1,10) and round(x,10) < round(segment.x2,10):
                        
                        return (segment.deflection(x - segment.x1)) - (dzi + (dzj-dzi)/L*x)
                    
                if isclose(x, self.L()):
                    
                    lastIndex = len(self.SegmentsY) - 1
                    return (self.SegmentsY[lastIndex].deflection(x - self.SegmentsY[lastIndex].x1)) - dzj
        
        else:

            return 0

    def plot_rel_deflection(self, Direction, combo_name='Combo 1', n_points=20):
        """
        Plots the deflection diagram for the member
        
        Parameters
        ----------
        Direction : {'dy', 'dz'}
            The direction in which to plot the deflection.
        combo_name : string
            The name of the load combination to get the results for (not the combination itself).
        """
        
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]
                
        # Import 'pyplot' if not already done
        if Member3D.__plt is None:
            from matplotlib import pyplot as plt
            Member3D.__plt = plt
        
        fig, ax = Member3D.__plt.subplots()
        ax.axhline(0, color='black', lw=1)
        ax.grid()
        
        x, d_relative = self.rel_deflection_array(Direction, n_points, combo_name)

        Member3D.__plt.plot(x, d_relative)
        Member3D.__plt.ylabel('Relative Deflection')
        Member3D.__plt.xlabel('Location')
        Member3D.__plt.title('Member ' + self.name + '\n' + combo_name)
        Member3D.__plt.show()

    def rel_deflection_array(self, Direction, n_points, combo_name='Combo 1', x_array=None):
        """
        Returns the array of the relative deflection in the member for the given direction
        
        Parameters
        ----------
        Direction : string
            The direction in which to find the relative deflection. Must be one of the following:
                'dy' = Deflection in the local y-axis
                'dz' = Deflection in the local z-axis
        n_points: int
            The number of points in the array to generate over the full length of the member.
        combo_name : string
            The name of the load combination to get the results for (not the load combination itself).
        x_array : array = None
            A custom array of x values that may be provided by the user, otherwise an array is generated.
            Values must be provided in local member coordinates (between 0 and L) and be in ascending order.
        """
        # Segment the member if necessary
        if self._solved_combo is None or combo_name != self._solved_combo.name:
            self._segment_member(combo_name)
            self._solved_combo = self.model.load_combos[combo_name]

        d = self.d(combo_name)
        dyi = d[1,0]
        dyj = d[7,0]
        dzi = d[2,0]
        dzj = d[8,0]

        L = self.L()
        if x_array is None:
            x_array = linspace(0, L, n_points)
        else:
            if any(x_array<0) or any(x_array>L):
                raise ValueError(f"All x values must be in the range 0 to {L}")

        # Check which axis is of interest
        if Direction == 'dy':
            deflections = self._extract_vector_results(self.SegmentsZ, x_array, 'deflection')[1]
            return vstack((x_array, deflections - (dyi + (dyj-dyi)/L*x_array)))
        
        elif Direction == 'dz':
            deflections = self._extract_vector_results(self.SegmentsY, x_array, 'deflection')[1]
            return vstack((x_array, deflections - (dzi + (dzj-dzi)/L*x_array)))        
        
    def _segment_member(self, combo_name='Combo 1'):
        """
        Divides the element up into mathematically continuous segments along each axis
        """
        
        # Get the member's length and stiffness properties
        L = self.L()
        E = self.material.E
        A = self.section.A
        Iz = self.section.Iz
        Iy = self.section.Iy
        SegmentsZ = self.SegmentsZ
        SegmentsY = self.SegmentsY
        SegmentsX = self.SegmentsX
        
        # Get the load combination to segment the member for
        combo = self.model.load_combos[combo_name]

        # Create a list of discontinuity locations
        disconts = [0, L] # Member ends
        
        for load in self.PtLoads: 
            disconts.append(load[2]) # Point load locations
        
        for load in self.DistLoads: 
            disconts.append(load[3]) # Distributed load start locations
            disconts.append(load[4]) # Distributed load end locations
        
        # Sort the list and eliminate duplicate values
        disconts = sorted(set(disconts))
        
        # Clear out old data from any previous analyses
        SegmentsZ.clear()
        SegmentsY.clear()
        SegmentsX.clear()
        
        # Create a list of mathematically continuous segments for each direction
        for index in range(len(disconts) - 1):
            
            # z-direction segments (bending about local z-axis)
            newSeg = BeamSegZ()           # Create the new segment
            newSeg.x1 = disconts[index]   # Segment start location
            newSeg.x2 = disconts[index+1] # Segment end location
            newSeg.EI = E*Iz              # Segment flexural stiffness
            newSeg.EA = E*A               # Segment axial stiffness
            SegmentsZ.append(newSeg)      # Add the segment to the list
            
            # y-direction segments (bending about local y-axis)
            newSeg = BeamSegY()           # Create the new segment
            newSeg.x1 = disconts[index]   # Segment start location
            newSeg.x2 = disconts[index+1] # Segment end location
            newSeg.EI = E*Iy              # Segment flexural stiffness
            newSeg.EA = E*A               # Segment axial stiffness
            SegmentsY.append(newSeg)      # Add the segment to the list
            
            # x-direction segments (for torsional moment)
            newSeg = BeamSegZ()           # Create the new segment
            newSeg.x1 = disconts[index]   # Segment start location
            newSeg.x2 = disconts[index+1] # Segment end location
            newSeg.EA = E*A               # Segment axial stiffness
            SegmentsX.append(newSeg)      # Add the segment to the list
        
        # Get the element local end forces, local fixed end reactions, and local displacements
        f = self.f(combo_name)           # Member local end force vector
        fer = self._fer_unc(combo_name)  # Member local fixed end reaction vector
        d = self.d(combo_name)           # Member local displacement vector
        
        # Get the local deflections and calculate the slope at the start of the member
        # Note 1: The slope may not be available directly from the local displacement vector if member end releases have been used, so slope-deflection has been applied to solve for it.
        # Note 2: The traditional slope-deflection equations assume a sign convention opposite of what Pynite uses for moments about the local y-axis, so a negative value has been applied to those values specifically.
        m1z = f[5, 0]       # local z-axis moment at start of member
        m2z = f[11, 0]      # local z-axis moment at end of member
        m1y = -f[4, 0]      # local y-axis moment at start of member
        m2y = -f[10, 0]     # local y-axis moment at end of member
        fem1z = fer[5, 0]   # local z-axis fixed end moment at start of member
        fem2z = fer[11, 0]  # local z-axis fixed end moment at end of member
        fem1y = -fer[4, 0]  # local y-axis fixed end moment at start of member
        fem2y = -fer[10, 0] # local y-axis fixed end moment at end of member
        delta1y = d[1, 0]   # local y displacement at start of member
        delta2y = d[7, 0]   # local y displacement at end of member
        delta1z = d[2, 0]   # local z displacement at start of member
        delta2z = d[8, 0]   # local z displacement at end of member
        SegmentsZ[0].delta1 = delta1y
        SegmentsY[0].delta1 = delta1z
        SegmentsZ[0].theta1 = 1/3*((m1z - fem1z)*L/(E*Iz) - (m2z - fem2z)*L/(2*E*Iz) + 3*(delta2y - delta1y)/L)
        SegmentsY[0].theta1 = -1/3*((m1y - fem1y)*L/(E*Iy) - (m2y - fem2y)*L/(2*E*Iy) + 3*(delta2z - delta1z)/L)

        # Add the axial deflection at the start of the member
        SegmentsZ[0].delta_x1 = d[0, 0]
        SegmentsY[0].delta_x1 = d[0, 0]
        SegmentsX[0].delta_x1 = d[0, 0]
        
        # Add loads to each segment
        for i in range(len(SegmentsZ)):
            
            # Get the starting point of the segment
            x = SegmentsZ[i].x1
            
            # Initialize the distributed loads on the segment to zero
            SegmentsZ[i].w1 = 0
            SegmentsZ[i].w2 = 0
            SegmentsZ[i].p1 = 0
            SegmentsZ[i].p2 = 0
            SegmentsY[i].w1 = 0
            SegmentsY[i].w2 = 0
            SegmentsY[i].p1 = 0
            SegmentsY[i].p2 = 0
            
            # Initialize the slope and displacement at the start of the segment
            if i > 0: # The first segment has already been initialized
                SegmentsZ[i].theta1 = SegmentsZ[i-1].slope(SegmentsZ[i-1].Length())
                SegmentsZ[i].delta1 = SegmentsZ[i-1].deflection(SegmentsZ[i-1].Length())
                SegmentsZ[i].delta_x1 = SegmentsZ[i-1].AxialDeflection(SegmentsZ[i-1].Length())
                SegmentsY[i].theta1 = SegmentsY[i-1].slope(SegmentsY[i-1].Length())
                SegmentsY[i].delta1 = SegmentsY[i-1].deflection(SegmentsY[i-1].Length())
                SegmentsY[i].delta_x1 = SegmentsY[i-1].AxialDeflection(SegmentsY[i-1].Length())
                
            # Add the effects of the beam end forces to the segment
            SegmentsZ[i].P1 = f[0, 0]
            SegmentsZ[i].V1 = f[1, 0]
            SegmentsZ[i].M1 = f[5, 0] - f[1, 0]*x
            SegmentsY[i].P1 = f[0, 0]
            SegmentsY[i].V1 = f[2, 0]
            SegmentsY[i].M1 = f[4, 0] + f[2, 0]*x
            SegmentsX[i].T1 = f[3, 0]
            
            # Step through each load case in the specified load combination
            for case, factor in combo.factors.items():
            
                # Add effects of point loads occuring prior to this segment
                for ptLoad in self.PtLoads:
                    
                    if round(ptLoad[2], 10) <= round(x, 10) and case == ptLoad[3]:
                    
                        if ptLoad[0] == 'Fx':
                            SegmentsZ[i].P1 += factor*ptLoad[1]
                        elif ptLoad[0] == 'Fy':
                            SegmentsZ[i].V1 += factor*ptLoad[1]
                            SegmentsZ[i].M1 -= factor*ptLoad[1]*(x - ptLoad[2])
                        elif ptLoad[0] == 'Fz':
                            SegmentsY[i].V1 += factor*ptLoad[1]
                            SegmentsY[i].M1 += factor*ptLoad[1]*(x - ptLoad[2])
                        elif ptLoad[0] == 'Mx':
                            SegmentsX[i].T1 += factor*ptLoad[1]    
                        elif ptLoad[0] == 'My':
                            SegmentsY[i].M1 += factor*ptLoad[1]
                        elif ptLoad[0] == 'Mz':
                            SegmentsZ[i].M1 += factor*ptLoad[1]
                        elif ptLoad[0] == 'FX' or ptLoad[0] == 'FY' or ptLoad[0] == 'FZ':
                            FX, FY, FZ = 0, 0, 0
                            if ptLoad[0] == 'FX': FX = 1
                            if ptLoad[0] == 'FY': FY = 1
                            if ptLoad[0] == 'FZ': FZ = 1
                            force = self.T()[:3, :][:, :3] @ array([FX*ptLoad[1], FY*ptLoad[1], FZ*ptLoad[1]])
                            SegmentsZ[i].P1 += factor*force[0]
                            SegmentsZ[i].V1 += factor*force[1]
                            SegmentsZ[i].M1 -= factor*force[1]*(x - ptLoad[2])
                            SegmentsY[i].V1 += factor*force[2]
                            SegmentsY[i].M1 += factor*force[2]*(x - ptLoad[2])
                        elif ptLoad[0] == 'MX' or ptLoad[0] == 'MY' or ptLoad[0] == 'MZ':
                            MX, MY, MZ = 0, 0, 0
                            if ptLoad[0] == 'MX': MX = 1
                            if ptLoad[0] == 'MY': MY = 1
                            if ptLoad[0] == 'MZ': MZ = 1
                            force = self.T()[:3, :][:, :3] @ array([MX*ptLoad[1], MY*ptLoad[1], MZ*ptLoad[1]])
                            SegmentsX[i].T1 += factor*force[0]
                            SegmentsY[i].M1 += factor*force[1]
                            SegmentsZ[i].M1 += factor*force[2]
                
                # Add distributed loads to the segment
                for distLoad in self.DistLoads:
                    
                    if case == distLoad[5]:
                    
                        # Get the parameters for the distributed load
                        Direction = distLoad[0]
                        w1 = factor*distLoad[1]
                        w2 = factor*distLoad[2]
                        x1 = distLoad[3]
                        x2 = distLoad[4]
            
                        # Determine if the load affects the segment
                        if round(x1, 10) <= round(x, 10):
                    
                            if Direction == 'Fx':
                        
                                # Determine if the load is on this segment
                                if round(x2,10) > round(x,10):
                                                
                                    # Break up the load and place it on the segment
                                    # Note that 'w1' and 'w2' are really the axial loads 'p1' and 'p2' here
                                    SegmentsZ[i].p1 += (w2 - w1)/(x2 - x1)*(x - x1) + w1
                                    SegmentsZ[i].p2 += (w2 - w1)/(x2 - x1)*(SegmentsZ[i].x2 - x1) + w1
                                    SegmentsY[i].p1 += (w2 - w1)/(x2 - x1)*(x - x1) + w1
                                    SegmentsY[i].p2 += (w2 - w1)/(x2 - x1)*(SegmentsY[i].x2 - x1) + w1
                            
                                    # Calculate the magnitude of the load at the start of the segment
                                    w2 = w1+(w2-w1)/(x2-x1)*(x-x1)
                                    x2 = x
                        
                                # Calculate the axial force at the start of the segment
                                SegmentsZ[i].P1 += (w1 + w2)/2*(x2 - x1)
                                SegmentsY[i].P1 += (w1 + w2)/2*(x2 - x1)
                    
                            elif Direction == 'Fy':
                        
                                # Determine if the load is on this segment
                                if round(x2,10) > round(x,10):
                                                
                                    # Break up the load and place it on the segment
                                    SegmentsZ[i].w1 += (w2 - w1)/(x2 - x1)*(x - x1) + w1
                                    SegmentsZ[i].w2 += (w2 - w1)/(x2 - x1)*(SegmentsZ[i].x2 - x1) + w1
                            
                                    # Calculate the magnitude of the load at the start of the segment
                                    # This will be used as the 'x2' value for the load prior to the start of the segment
                                    w2 = w1 + (w2 - w1)/(x2 - x1)*(x - x1)
                                    x2 = x
                        
                                # Calculate the shear and moment at the start of the segment due to the load
                                SegmentsZ[i].V1 += (w1 + w2)/2*(x2 - x1)
                                SegmentsZ[i].M1 -= (x1 - x2)*(2*w1*x1 - 3*w1*x + w1*x2 + w2*x1 - 3*w2*x + 2*w2*x2)/6
                    
                            elif Direction == 'Fz':
                        
                                # Determine if the load is on this segment
                                if round(x2,10) > round(x,10):
                                                
                                    # Break up the load and place it on the segment
                                    SegmentsY[i].w1 += (w2 - w1)/(x2 - x1)*(SegmentsY[i].x1 - x1) + w1
                                    SegmentsY[i].w2 += (w2 - w1)/(x2 - x1)*(SegmentsY[i].x2 - x1) + w1
                            
                                    # Calculate the magnitude of the load at the start of the segment
                                    w2 = w1 + (w2 - w1)/(x2 - x1)*(x - x1)
                                    x2 = x
                        
                                # Calculate the shear and moment at the start of the segment due to the load
                                SegmentsY[i].V1 += (w1 + w2)/2*(x2 - x1)
                                SegmentsY[i].M1 += (x1 - x2)*(2*w1*x1 - 3*w1*x + w1*x2 + w2*x1 - 3*w2*x + 2*w2*x2)/6

                            elif Direction == 'FX' or Direction == 'FY' or Direction == 'FZ':
                                
                                FX, FY, FZ = 0, 0, 0
                                if Direction == 'FX': FX = 1
                                if Direction == 'FY': FY = 1
                                if Direction == 'FZ': FZ = 1
                                T = self.T()[:3, :][:, :3]
                                f1 = T @ array([FX*w1, FY*w1, FZ*w1])
                                f2 = T @ array([FX*w2, FY*w2, FZ*w2])

                                # Determine if the load is on this segment
                                if round(x2, 10) > round(x, 10):
                                                
                                    # Break up the load and place it on the segment
                                    SegmentsZ[i].p1 += (f2[0] - f1[0])/(x2 - x1)*(x - x1) + f1[0]
                                    SegmentsZ[i].p2 += (f2[0] - f1[0])/(x2 - x1)*(SegmentsZ[i].x2 - x1) + f1[0]
                                    SegmentsY[i].p1 += (f2[0] - f1[0])/(x2 - x1)*(x - x1) + f1[0]
                                    SegmentsY[i].p2 += (f2[0] - f1[0])/(x2 - x1)*(SegmentsY[i].x2 - x1) + f1[0]

                                    SegmentsZ[i].w1 += (f2[1] - f1[1])/(x2 - x1)*(x - x1) + f1[1]
                                    SegmentsZ[i].w2 += (f2[1] - f1[1])/(x2 - x1)*(SegmentsZ[i].x2 - x1) + f1[1]

                                    SegmentsY[i].w1 += (f2[2] - f1[2])/(x2 - x1)*(SegmentsY[i].x1 - x1) + f1[2]
                                    SegmentsY[i].w2 += (f2[2] - f1[2])/(x2 - x1)*(SegmentsY[i].x2 - x1) + f1[2]

                                    # Calculate the magnitude of the load at the start of the segment
                                    w2 = w1 + (w2 - w1)/(x2 - x1)*(x - x1)
                                    f2 = T @ array([FX*w2, FY*w2, FZ*w2])
                                    x2 = x

                                # Calculate the axial force, shear and moment at the start of the segment
                                SegmentsZ[i].P1 += (f1[0] + f2[0])/2*(x2 - x1)
                                SegmentsY[i].P1 += (f1[0] + f2[0])/2*(x2 - x1)

                                SegmentsZ[i].V1 += (f1[1] + f2[1])/2*(x2 - x1)
                                SegmentsZ[i].M1 -= (x1 - x2)*(2*f1[1]*x1 - 3*f1[1]*x + f1[1]*x2 + f2[1]*x1 - 3*f2[1]*x + 2*f2[1]*x2)/6

                                SegmentsY[i].V1 += (f1[2] + f2[2])/2*(x2 - x1)
                                SegmentsY[i].M1 += (x1 - x2)*(2*f1[2]*x1 - 3*f1[2]*x + f1[2]*x2 + f2[2]*x1 - 3*f2[2]*x + 2*f2[2]*x2)/6
                            
    def _extract_vector_results(self, segments, x_array, result_name, P_delta=False):
        """Extract results from the given segments using vectorised numpy functions"""

        # Initialize variables
        segment_results = []
        last_index = 0
        
        # Step through each segment in the member
        for i, segment in enumerate(segments):

            # Determine if the current index `i` is at the end of the member
            if i == len(segments) - 1:
                index2 = len(x_array)
            else:
                try:
                    index2 = where(x_array > segment.x2)[0][0]
                except IndexError:
                    index2 = len(x_array)
            
            if last_index == index2: continue

            thisseg_x_array = x_array[last_index:index2]
            
            if result_name == "moment":
                thisseg_y_array = segment.moment(thisseg_x_array - segment.x1, P_delta)
            elif result_name == "shear":
                thisseg_y_array = segment.Shear(thisseg_x_array - segment.x1)
            elif result_name == "axial":
                thisseg_y_array = segment.axial(thisseg_x_array - segment.x1)
            elif result_name == "torque":
                thisseg_y_array = segment.Torsion(thisseg_x_array - segment.x1)
            elif result_name == "deflection":
                thisseg_y_array = segment.deflection(thisseg_x_array - segment.x1, P_delta)
            elif result_name == "axial_deflection":
                thisseg_y_array = segment.AxialDeflection(thisseg_x_array - segment.x1)

            segment_results.append(thisseg_y_array)

            last_index = index2

        return vstack((x_array, hstack(segment_results)))