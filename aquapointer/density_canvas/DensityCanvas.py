import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable
from typing import Union
from typing_extensions import Self
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numbers
import math
import Lp_norm as lpn
import embedding as emb
from itertools import combinations

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
        self._density = slice
        self._empty = False
        self._density_type = "data"
    
    def set_density_from_gaussians(self, centers: ArrayLike, amplitude: numbers.Number, variance: numbers.Number):
        self.clear_density()
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
    
    def clear_lattice(self):
        try:
            del self._lattice
            del self._min_spacing
            del self._lattice_type
        except AttributeError:
            pass
        
    def define_custom_lattice(self, pos: ArrayLike):
        self._lattice = np.array(pos)
        self._min_spacing = emb.find_minimal_distance(pos)
        self._lattice_type = "custom"

    def define_rectangular_lattice(self, xnum: int, ynum: int, spacing: numbers.Number, center: ArrayLike = None):
        if not center:
            center = (
                self._length_x/2 + self._origin[0], 
                self._length_y/2 + self._origin[1], 
            )
        
        xlength = (xnum-1)*spacing
        ylength = (ynum-1)*spacing

        x_coords = []
        for i in range(xnum):
            x_coords.append(center[0] - (xlength/2) + i*spacing)
        
        y_coords = []
        for i in range(ynum):
            y_coords.append(center[1] - (ylength/2) + i*spacing)

        coords = []
        for xc in x_coords:
            for yc in y_coords:
                coords.append([xc, yc])
        
        self._lattice = np.array(coords)
        self._min_spacing = spacing
        self._lattice_type = "rectangular"

    def calculate_ubo_coefficients(self, p: int, params: ArrayLike, high=None, low=None):
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
            print("Lattice needs to be defined beforehand")
            return
        
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
            stg.set_density_from_gaussians([lattice[i]], *pms)
            return stg

        # calculate using formula
        self._ubo = {
            "coeffs": lpn.Lp_coefficients(len(lattice), p, _base, _component, params, high, low),
            "p": p,
            "params": params,
            "high": high,
            "low": low
        }
    
    def calculate_linear_coefficients(self, p: int, params: ArrayLike):
        """ Calcuates the linear coefficients of the cost function.
        If ubo coefficients have already been calculated, fetch them.
        Otherwise use formula and return coefficient list
        """

        # try to fetch linear coefficients from existing ubo
        try:
            coeffs = self._ubo["coeffs"]
            self._linear = {
                "gammas": list(coeffs[1].values()),
                "p": self._ubo["p"],
                "params": self._ubo["params"]
            }
            return
        except AttributeError:
            pass
        except KeyError:
            pass
        
        # check that lattice exists
        try:
            lattice = self._lattice
        except AttributeError:
            print("Lattice needs to be defined beforehand")
            return
        
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
            stg.set_density_from_gaussians([lattice[i]], *pms)
            return stg

        # calculate using formula
        self._linear = {
            "gammas": lpn.Lp_coefficients_first_order(len(lattice), p, _base, _component, params),
            "p": p,
            "params": params,
        }
    
    def clear_ubo(self):
        try:
            del self._ubo
        except AttributeError:
            pass
    
    def clear_linear(self):
        try:
            del self._linear
        except AttributeError:
            pass
    
    def calculate_bitstring_cost_from_coefficients(self, bitstring: Union[str, ArrayLike], high=None, low=None) -> numbers.Number:
        try: 
            coeffs = self._ubo["coeffs"]
        except AttributeError:
            print("Coefficients need to be computed beforehand")
            return
        
        # set high and low
        if high is None:
            high = self._ubo["high"]
        if low is None:
            low = self._ubo["low"]
        if high > self._ubo["high"]:
            print(f"Warning: calculating up to maximum degree {self._ubo['high']}")
            high = self._ubo["high"]
        if low < self._ubo["low"]:
            print(f"Warning: calculating up to minimum degree {self._ubo['low']}")
            low = self._ubo["low"]
        
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
    
    def calculate_bitstring_cost_from_norm(self, bitstring: Union[str, ArrayLike], mixture_params: ArrayLike, norm: Callable, norm_params: ArrayLike) -> numbers.Number:
        try:
            lattice = self._lattice
        except AttributeError:
            print("Lattice needs to be defined beforehand")
            return
        
        # separate bitstring in list of binaries
        bits = np.array([int(b) for b in bitstring])
        
        # calculate cost
        candidate_centers = [lattice[i] for i in range(len(lattice)) if bits[i]==1]
        test = DensityCanvas(self._origin, self._length_x, self._length_y, self._npoints_x, self._npoints_y)
        test.set_density_from_gaussians(candidate_centers, *mixture_params)
        return norm(self-test, *norm_params) - norm(self, *norm_params)
            
    def plotting_objects(self, figsize=(10,8), draw_centers=False, draw_lattice=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_box_aspect(self._length_y/self._length_x)
        c = ax.pcolormesh(self._X, self._Y, self._density, cmap='viridis')
        if draw_lattice:
            try:
                ax.scatter(self._lattice[:,0], self._lattice[:,1], color='blue')
            except AttributeError:
                print("Lattice has not been defined")
                return
        if draw_centers:
            try:
                ax.scatter(self._centers[:,0], self._centers[:,1], color='red', marker="x", s=120)
            except AttributeError:
                print("Centers have not been defined")
                return
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        fig.colorbar(c)
        return fig, ax
    
    def draw(self, figsize=(10,8), draw_centers=False, draw_lattice=False):
        fig, ax = self.plotting_objects(figsize, draw_centers, draw_lattice)
        plt.show()