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
from aquapointer.density_canvas.Lattice import Lattice

class DensityCanvas:
    """ This is a class that contains information on 2D slice-space.
    It allows to handle operations involving densities, including
    calculating register embeddings and qubo coefficients"""

    def __init__(
        self,
        origin: ArrayLike,
        length_x: float,
        length_y: float,
        npoints_x: int,
        npoints_y: int,
    ):
        self._origin = np.array(origin)
        
        self._length_x = length_x
        self._length_y = length_y
        
        self._npoints_x = npoints_x
        self._npoints_y = npoints_y
        
        self._fill_derived_attributes()

    def _fill_derived_attributes(self):
        # min and max coordinates
        self._min_x = self._origin[0]
        self._max_x = self._min_x + self._length_x
        self._min_y = self._origin[1]
        self._max_y = self._min_y + self._length_y

        # center
        self._center = np.array(
            [self._length_x/2+self._origin[0], self._length_y/2+self._origin[1]]
        )
        
        # unit length and unit area
        self._shape = (self._npoints_y, self._npoints_x)
        self._dx = self._length_x/self._npoints_x
        self._dy = self._length_y/self._npoints_y
        self._dA = self._dx*self._dy
        
        # the density is initialized as an array of zeros
        self._density = np.zeros(self._shape)
        self._empty = True
        self._density_type = None
        
        # meshgrid shenenigans
        self._array_x = np.linspace(self._min_x, self._max_x, self._npoints_x)
        self._array_y = np.linspace(self._min_y, self._max_y, self._npoints_y)
        self._X, self._Y = np.meshgrid(self._array_x, self._array_y)
        self._pos = np.empty(self._X.shape + (2,))
        self._pos[:, :, 0] = self._X
        self._pos[:, :, 1] = self._Y

    def __add__(self, other: Self) -> Self:
        if np.linalg.norm(self._origin-other._origin) > 1e-8:
            raise ValueError("The two canvases need to have the same origin")
        if np.abs(self._length_x - other._length_x) > 1e-8:
            raise ValueError("The two canvases need to have the same length on the x-axis")
        if np.abs(self._length_y - other._length_y) > 1e-8:
            raise ValueError("The two canvases need to have the same length on the y-axis")
        if self._npoints_x != other._npoints_x:
            raise ValueError("The two canvases need to have the same number of points on the x-axis")
        if self._npoints_y != other._npoints_y:
            raise ValueError("The two canvases need to have the same number of points on the y-axis")
        
        res = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        res._density = self._density + other._density
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __sub__(self, other: Self) -> Self:
        if np.linalg.norm(self._origin-other._origin) > 1e-8:
            raise ValueError("The two canvases need to have the same origin")
        if np.abs(self._length_x - other._length_x) > 1e-8:
            raise ValueError("The two canvases need to have the same length on the x-axis")
        if np.abs(self._length_y - other._length_y) > 1e-8:
            raise ValueError("The two canvases need to have the same length on the y-axis")
        if self._npoints_x != other._npoints_x:
            raise ValueError("The two canvases need to have the same number of points on the x-axis")
        if self._npoints_y != other._npoints_y:
            raise ValueError("The two canvases need to have the same number of points on the y-axis")
        
        res = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        res._density = self._density - other._density
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __mul__(self, other: Union[Self, numbers.Number]) -> Self:
        if isinstance(other, numbers.Number):
            res = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
            if self._empty:
                return res
            res._density = self._density * other
            res._empty = False
            res._density_type = "arithmetic"
            return res

        if np.linalg.norm(self._origin-other._origin) > 1e-8:
            raise ValueError("The two canvases need to have the same origin")
        if np.abs(self._length_x - other._length_x) > 1e-8:
            raise ValueError("The two canvases need to have the same length on the x-axis")
        if np.abs(self._length_y - other._length_y) > 1e-8:
            raise ValueError("The two canvases need to have the same length on the y-axis")
        if self._npoints_x != other._npoints_x:
            raise ValueError("The two canvases need to have the same number of points on the x-axis")
        if self._npoints_y != other._npoints_y:
            raise ValueError("The two canvases need to have the same number of points on the y-axis")
        
        res = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        if self._empty or other._empty:
            return res
        res._density = self._density * other._density
        res._empty = False
        res._density_type = "arithmetic"
        return res
    
    def __rmul__(self, other: Self) -> Self:
        return self * other
    
    def __pow__(self, other: numbers.Number) -> Self:
        res = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        res._density = np.power(self._density, other)
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __abs__(self) -> Self:
        res = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        res._density = np.abs(self._density)
        res._empty = self._empty
        res._density_type = self._density_type
        try:
            res._centers = self._centers
            res._amplitude = self._amplitude
            res._variance = self._variance
        except AttributeError:
            pass
        return res
    
    def __float__(self) -> numbers.Number:
        return float(self.integrate())

    def integrate(self) -> numbers.Number:
        if self._empty:
            return 0
        return np.sum(self._density)*self._dA

    def clear_density(self):
        self._density = np.zeros(self._density.shape)
        self._empty = True
        self._density_type = None
        try:
            del self._centers
            del self._variance
            del self._amplitude
        except AttributeError:
            pass

    def set_density_from_slice(self, slice: ArrayLike):
        if slice.shape != self._density.shape:
            raise ValueError(f"The slice must have shape {self._density.shape}")
        self.clear_density()
        self.clear_pubo()
        self._density = slice
        self._empty = False
        self._density_type = "data"
    
    def set_density_from_gaussians(self, centers: ArrayLike, amplitude: numbers.Number, variance: numbers.Number):
        self.clear_density()
        self.clear_pubo()
        rvs = []
        for mu in centers:
            rvs.append(multivariate_normal(mu, variance))
        for rv in rvs:
            self._density += amplitude*rv.pdf(self._pos)
        self._empty = False
        self._centers = np.array([np.array(c) for c in centers])
        self._variance = variance
        self._amplitude = amplitude
        self._density_type = "gaussians"
    
    def set_randomized_gaussian_density(self, n_centers: numbers.Number, amplitude: numbers.Number, variance: numbers.Number, minimal_distance: numbers.Number, padding: numbers.Number, seed: numbers.Number = None):
        np.random.seed(seed)
        centers = []
        count = 0
        while (len(centers) < n_centers) and count<10000:
            x = np.random.rand()*(self._length_x-2*padding)+(self._origin[0]+padding)
            y = np.random.rand()*(self._length_y-2*padding)+(self._origin[1]+padding)
            c = np.array([x,y])
            check = True
            for c1 in centers:
                if np.linalg.norm(c-c1) < minimal_distance:
                    check = False
                    count += 1
                    break
            if check:
                centers.append(c)
        if len(centers)<n_centers:
            raise ValueError("It was not possible to generate that many Gaussians on this canvas")
        self.set_density_from_gaussians(centers, amplitude, variance)

    def set_lattice(self, lattice: Lattice, centering=True):
        self.clear_lattice()
        self.clear_pubo()
        if self._length_x < lattice._length_x:
            raise ValueError("The lattice does not fit in the canvans along the x direction")
        if self._length_y < lattice._length_y:
            raise ValueError("The lattice does not fit in the canvans along the y direction")
        self._lattice = lattice
        
        shift = self._center - lattice._center if centering else np.zeros(2)
        self._lattice._coords += shift
        
        try:
            self._lattice._history = [np.array(h)+shift for h in self._lattice._history]
        except AttributeError:
            pass

    def set_rectangular_lattice(self, num_x, num_y, spacing):
        lattice = Lattice.rectangular(num_x=num_x, num_y=num_y, spacing=spacing)
        self.set_lattice(lattice, centering=True)

    def set_poisson_disk_lattice(self, spacing: tuple):
        lattice = Lattice.poisson_disk(
            density=self._density,
            length=(self._length_x, self._length_y),
            spacing=spacing,
        )
        self.set_lattice(lattice, centering=True)
    
    def clear_lattice(self):
        try:
            del self._lattice
        except AttributeError:
            pass

    def lattice_dynamics(self, spacing: numbers.Number, T: numbers.Number = 1000, dt: numbers.Number = 1, save_history=False, viscosity=0.1, centering=True):
        self._lattice._coords -= self._center - self._lattice._center if centering else np.zeros(2)
        self._lattice.dynamics(self._density, (self._length_x, self._length_y), spacing, T, dt, save_history, viscosity)
        self._lattice._coords += self._center - self._lattice._center if centering else np.zeros(2)
        if save_history:
            for h in self._lattice._history:
                for c in h:
                    c += self._center - self._lattice._center if centering else np.zeros(2)
        try:
            self.calculate_pubo_coefficients(
                p=self._pubo["p"],
                params=self._pubo["params"],
                high=self._pubo["high"],
                low=self._pubo["low"],
            )
        except AttributeError:
            pass
        
    def calculate_pubo_coefficients(self, p: int, params: ArrayLike, high=None, low=None):
        """ Calcuates the coefficients of the cost function.
        The coefficients are stored in a dictionary {1:{}, 2:{}, ..., p:{}} where the key
        represents the interaction order and the values are dictionaries.
        The dictionary associated to a key is of the type {(0,1): val, (0,2): val, ...}
        where each tuple represents the index of the variables that take part
        in the interaction, and values are the interaction strength.
        Example: with 4 variables 0,1,2,3 and cost function of order p=2, you have:
        {
            1: {(0,): val, (1,): val, (2,): val, (3,): val},
            2: {(0,1): val, (0,2): val, (0,3): val, (1,2): val, (1,3): val, (2,3): val}
        }
        """
        try:
            lattice = self._lattice
        except AttributeError:
            raise AttributeError("Lattice needs to be defined in order to calculate pubo coefficients")
        
        if high is None:
            high = p
        if low is None:
            low = 1

        if high > p:
            print(f"Warning: calculating up to maximum degree {p}")
            high = p
        if low < 1:
            print(f"Warning: calculating up to minimum degree {1}")
            low = 1

        # define base and component functions
        def _base() -> Self:
            return self
        def _component(i: int, pms: ArrayLike) -> Self:
            stg = DensityCanvas(
                origin=self._origin,
                length_x=self._length_x,
                length_y=self._length_y,
                npoints_x=self._npoints_x,
                npoints_y=self._npoints_y
            )
            stg.set_density_from_gaussians([lattice._coords[i]], *pms)
            return stg

        # calculate using formula
        self._pubo = {
            "coeffs": lpn.Lp_coefficients(len(lattice._coords), p, _base, _component, params, high, low),
            "p": p,
            "params": params,
            "high": high,
            "low": low
        }
    
    def clear_pubo(self):
        try:
            del self._pubo
        except AttributeError:
            pass
    
    def decimate_lattice(self):
        # check that lattice exists
        try:
            lattice = self._lattice
        except AttributeError:
            raise AttributeError("Lattice needs to be defined in order to decimate it")
        
        # check that coefficients have been calculated
        try:
            linear = self._pubo["coeffs"][1]
        except AttributeError:
            raise AttributeError("Coefficients need to be calculated before decimation")
        except KeyError:
            raise AttributeError("Linear coefficients need to be calculated before decimation")
        
        new_coords = [c for i,c in enumerate(lattice._coords) if linear[(i,)] < 0]
        new_lattice = Lattice(
            np.array(new_coords),
            length_x = lattice._length_x,
            length_y = lattice._length_y,
            center = lattice._center,
            lattice_type=f'{lattice._lattice_type}(decimated)'
        )
        pubo = self._pubo
        self.set_lattice(new_lattice, centering=False)
        self.calculate_pubo_coefficients(
            p=pubo["p"],
            params=pubo["params"],
            high=pubo["high"],
            low=pubo["low"],
        )
    
    def calculate_bitstring_cost_from_coefficients(self, bitstring: Union[str, ArrayLike], high=None, low=None) -> numbers.Number:
        try: 
            coeffs = self._pubo["coeffs"]
        except AttributeError:
            raise AttributeError("Pubo coefficients need to be computed in order to calculate cost")
        
        # set high and low
        if high is None:
            high = self._pubo["high"]
        if low is None:
            low = self._pubo["low"]
        if high > self._pubo["high"]:
            print(f"Warning: calculating up to maximum degree {self._pubo['high']}")
            high = self._pubo["high"]
        if low < self._pubo["low"]:
            print(f"Warning: calculating up to minimum degree {self._pubo['low']}")
            low = self._pubo["low"]
        
        # separate bitstring in list of binaries
        bits = np.array([int(b) for b in bitstring])

        # calculate cost
        cost = 0
        for i in range(low, high+1):
            for idx in coeffs[i]:
                if np.any(bits[list(idx)] == 0):
                    continue
                cost += coeffs[i][idx]
        return cost
    
    def calculate_bitstring_cost_from_distance(self, bitstring: Union[str, ArrayLike], mixture_params: ArrayLike, distance: Callable, **kwargs) -> numbers.Number:
        try:
            lattice = self._lattice
        except AttributeError:
            raise AttributeError("Lattice needs to be defined in order to calculate cost")
        
        # separate bitstring in list of binaries
        bits = np.array([int(b) for b in bitstring])
        
        # calculate cost
        candidate_centers = [lattice._coords[i] for i in range(len(lattice._coords)) if bits[i]==1]
        test = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        test.set_density_from_gaussians(candidate_centers, *mixture_params)
        return distance(self, test, **kwargs)
            
    def plotting_objects(self, figsize=(10,8), draw_centers=False, draw_lattice=False, lattice_history=False, labels=True, draw_connections=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_box_aspect(self._length_y/self._length_x)
        c = ax.pcolormesh(self._X, self._Y, self._density, cmap='viridis')
        if draw_lattice:
            try:
                ax.scatter(self._lattice._coords[:,0], self._lattice._coords[:,1], color='tab:orange')
                if labels:
                    yshift = self._length_y/50
                    for i,cd in enumerate(self._lattice._coords):
                        plt.text(cd[0], cd[1]-yshift, str(i))
            except AttributeError:
                print("Lattice has not been defined")
        if lattice_history:
            try:
                for trajectory in self._lattice._history:
                    ax.plot(np.array(trajectory)[:,0], np.array(trajectory)[:,1], color='tab:orange')
            except AttributeError:
                print("no history for this lattice")
        if draw_centers:
            try:
                ax.scatter(self._centers[:,0], self._centers[:,1], color='red', marker="x", s=120)
            except AttributeError:
                print("Centers have not been defined")
        if draw_connections:
            try:
                coords = self._lattice._coords
                n = len(coords)
                ref = np.min(np.abs(list(self._pubo["coeffs"][1].values())))
                interaction = self._pubo["coeffs"][2]
                for i in range(n):
                    for j in range(i+1, n):
                        if interaction[(i,j)] > ref:
                            ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], color='k', alpha=.2)
            except AttributeError:
                print("Can't draw connections without PUBO coefficients")
            

        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        fig.colorbar(c)
        return fig, ax
    
    def draw(self, figsize=(10,8), draw_centers=False, draw_lattice=False, lattice_history=False, labels=True, draw_connections=False):
        fig, ax = self.plotting_objects(figsize, draw_centers, draw_lattice, lattice_history, labels, draw_connections)
        plt.show()