import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable
from typing import Union
from typing_extensions import Self
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numbers
import math
import aquapointer.density_canvas.Lp_norm as lpn
import aquapointer.density_canvas.embedding as emb
from itertools import combinations
from scipy.stats import qmc
from scipy.integrate import RK45

class Lattice:
    """ This is a class that contains information on 2D lattices"""

    def __init__(
        self,
        coords: ArrayLike,
        **kwargs
    ):
        self._coords = np.array(coords, dtype=float)
        try:
            self._min_spacing = kwargs['min_distance']
        except KeyError:
            self._min_spacing = emb.find_minimal_distance(self._coords)
        try:
            self._length_x = kwargs['length_x']
        except KeyError:
            self._length_x = emb.find_maximal_distance(self._coords[:,0])
        try:
            self._length_y = kwargs['length_y']
        except KeyError:
            self._length_y = emb.find_maximal_distance(self._coords[:,1])
        try:
            self._center = kwargs['center']
        except KeyError:
            self._center = np.array([self._length_x/2, self._length_y/2])
        try: 
            self.rotation = kwargs['rotation']
        except KeyError:
            self.rotation = None
        try:
            self._lattice_type = kwargs['lattice_type']
        except KeyError:
            self._lattice_type = "custom"

    @classmethod
    def rectangular(cls, num_x: int, num_y: int, spacing: numbers.Number):
        length_x = (num_x-1)*spacing
        length_y = (num_y-1)*spacing
        
        x_coords = []
        for i in range(num_x):
            x_coords.append(i*spacing)
        
        y_coords = []
        for i in range(num_y):
            y_coords.append(i*spacing)

        coords = []
        for xc in x_coords:
            for yc in y_coords:
                coords.append([xc, yc])

        center = np.array([length_x/2, length_y/2])
       
        return cls(np.array(coords, dtype=float), min_spacing=spacing, length_x=length_x, length_y=length_y, center=center, type="rectangular")

    @classmethod
    def triangular(cls, nrows: int, ncols: int, spacing: numbers.Number):
        coords = np.mgrid[:ncols, :nrows].transpose().reshape(-1, 2).astype(float)
        coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
        coords[:, 1] *= np.sqrt(3) / 2
        coords *= spacing
        center = np.array([np.mean(coords[:,0]), np.mean(coords[:,1])])

        return cls(np.array(coords, dtype=float), min_spacing=spacing, center=center, type="triangular")
    
    @classmethod
    def hexagonal(cls, nrows: int, ncols: int, spacing: numbers.Number):
        rows = range(nrows)
        cols = range(ncols)
        xx = (0.5 + i + i // 2 + (j % 2) * ((i % 2) - 0.5) for i in cols for j in rows)
        h = np.sqrt(3) / 2
        yy = (h * j for i in cols for j in rows)
        
        coords = spacing*np.array([(x, y) for x, y in zip(xx, yy)])
        center = np.array([np.mean(coords[:,0]), np.mean(coords[:,1])])

        return cls(np.array(coords, dtype=float), min_spacing=spacing, center=center, type="hexagonal")

    @classmethod
    def poisson_disk(cls, density: ArrayLike, length: tuple, spacing: tuple, max_num: int = 8000):
        """
        Poisson disk sampling with variable radius.
        density: 2d array representing probability density
        length: tuple (length_x, length_y) representing the physical size of the 2d space
        spacing: tuple (min_radius, max_radius) representing the minimum and maximum exclusion radius
        max_num: maximum number of points to sample
        """
        
        min_radius, max_radius = spacing
        length_x, length_y = length
        scale_x, scale_y = np.array(length)/np.array(density.shape[::-1])

        def _index_from_position(pos):
            idx_x = int((pos[1])/scale_x)
            idx_y = int((pos[0])/scale_y)
            return (idx_x, idx_y)
        
        # convert probability density map to a radius map
        radius_density = -density #high probability = small radius and viceversa
        radius_density -= np.min(radius_density)
        radius_density *= (max_radius-min_radius)/np.max(radius_density)
        radius_density += min_radius
        # now radius_density is a 2d array with values between max_radius and min_radius

        coords = []
        queue = []
        num = 0

        # pick the first point as the density maximum and initialize queue
        first_point = np.array([np.random.rand()*length_x, np.random.rand()*length_y])
        coords.append(first_point)
        queue.append(num)
        num += 1

        # sample until max number is reached or points cannot be placed
        while len(queue) and (num<max_num):
            i = np.random.choice(queue)
            ref_point = coords[i]
            ref_radius = radius_density[_index_from_position(ref_point)]
            placed = False

            # try placement 30 times (default hard-coded value)
            tries = 0
            while (tries < 30) and not placed:
                # pick a random point at random distance between [ref_radius, 2*ref_radius]
                r = (np.random.rand()+1)*ref_radius
                theta = 2*np.pi*np.random.rand()
                new_point = np.array((r*np.cos(theta), r*np.sin(theta))) + ref_point

                # burn a try if point falls outside space 
                if not (0 <= new_point[0] < length_x):
                    continue
                if not (0 <= new_point[1] < length_y):
                    continue

                new_radius = radius_density[_index_from_position(new_point)]

                # now check if new_point is too close to some other point based on new_radius
                too_close = False
                for other_point in coords:
                    d = np.linalg.norm(new_point-other_point)
                    if d < new_radius:
                        too_close = True
                        break

                if too_close:
                    tries +=1
                else:
                    coords.append(new_point)
                    queue.append(num)
                    num += 1
                    placed = True

            if not placed:
                # after 30 tries, no point could be placed around i. remove it from the active queue
                queue.remove(i)

            center = np.array([length_x/2, length_y/2])
        
        return cls(np.array(coords, dtype=float), center=center, type="poisson_disk")


    def dynamics(self, density: ArrayLike, length: tuple, spacing: numbers.Number, T: numbers.Number = 1000, dt: numbers.Number = 1, save_history=False, viscosity=0.1):
        """ Calculates newtonian dynamics treating the lattice points as
        particles subject to Lennard-Jones interactions plus a space dependent
        scalar field given by -density.
        density: 2d array representing probability density, it is turned into attractive potential by putting a - sign in front 
        length: tuple (length_x, length_y) representing the physical size of the 2d space
        spacing: float indicating the minimal spacing between particles (minimum of Lennard-Jones potential)
        T: total time of the dynamics
        dt: timestep in integrator
        """

        def _leapfrog():
            t = 0
            while t<T:
                # half update velocities
                for i in range(len(vs)):
                    vs[i] += (_LennardJones_force(i) - _gradient(i) - _viscosity(i)) * dt/2
                    #vs[i] += -_gradient(i) * dt/2
                # update positions
                for i in range(len(xs)):
                    xs[i] += vs[i] * dt
                    if xs[i][0]>length[0]:
                        xs[i][0]=length[0]
                    if xs[i][0]<0:
                        xs[i][0]=0
                    if xs[i][1]>length[1]:
                        xs[i][1]=length[1]
                    if xs[i][1]<0:
                        xs[i][1]=0
                    history[i].append(np.array(xs[i]))
                # half update velocities
                for i in range(len(vs)):
                    vs[i] += (_LennardJones_force(i) - _gradient(i) - _viscosity(i)) * dt/2
                    #vs[i] += -_gradient(i) * dt/2
                t += dt

        def _viscosity(i):
            return K*vs[i]
        
        def _gradient(i):
            idx_x, idx_y = _index_from_position(xs[i])
            grad = np.gradient(attractive_field) 
            field_contribution = np.array([grad[1][idx_x, idx_y], grad[0][idx_x, idx_y]])
            return field_contribution
                
        def _index_from_position(pos):
            idx_x = min(int((pos[1])/scale_x), density.shape[0]-1)
            idx_y = min(int((pos[0])/scale_y), density.shape[1]-1)
            return (idx_x, idx_y)
        
        def _position_from_index(idx):
            return np.array([idx[1]*scale_x, idx[0]*scale_y])
        
        def _LennardJones_force(i):
            LJ_force = np.zeros(2)
            # sum contribution of all other particles
            for j, _ in enumerate(xs):
                if j==i:
                    continue
                r_ij = xs[i]-xs[j]
                r = max(np.linalg.norm(r_ij), 1e-1)
                # force is derivative of Lennard-Jones potential
                LJ_force += A*(2*(sigma**2/r**3)-(sigma/r**2)) * r_ij
            return LJ_force
        
        scale_x, scale_y = np.array(length)/np.array(density.shape)
        
        # define potential parameters
        attractive_field = -density
        sigma = 12 * spacing/2 #lennard-jones minimal distance is 2**(1/6) sigma
        A = 1e-4 * 4 * np.max(density) #the depth of the lennard-jones potential is A/4
        K = viscosity #viscosity constant
        
        # initialize positions and velocities
        xs = self._coords
        vs = np.zeros(xs.shape)
        #vs = .1*(2*np.random.rand(*xs.shape)-1)

        # initialize history        
        history = []
        for x in xs:
            history.append([np.array(x)])

        # compute dynamics
        _leapfrog()

        if save_history:
            self._history = history
        
        self._min_spacing = emb.find_minimal_distance(self._coords)
        self._length_x = emb.find_maximal_distance(self._coords[:,0])
        self._length_y = emb.find_maximal_distance(self._coords[:,1])
        self._lattice_type = "dynamics"



        
        

        











