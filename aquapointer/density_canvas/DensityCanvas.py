import array
import numbers
from collections.abc import Callable
from itertools import product
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal
from typing_extensions import Self

import aquapointer.density_canvas.embedding as emb
import aquapointer.density_canvas.Lp_norm as lpn
from aquapointer.density_canvas.Lattice import Lattice


class DensityCanvas:
    """This is a class that contains information on 2D slice-space.
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
            [self._length_x / 2 + self._origin[0], self._length_y / 2 + self._origin[1]]
        )

        # unit length and unit area
        self._shape = (self._npoints_y, self._npoints_x)
        self._dx = self._length_x / self._npoints_x
        self._dy = self._length_y / self._npoints_y
        self._dA = self._dx * self._dy

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
        if np.linalg.norm(self._origin - other._origin) > 1e-8:
            raise ValueError("The two canvases need to have the same origin")
        if np.abs(self._length_x - other._length_x) > 1e-8:
            raise ValueError(
                "The two canvases need to have the same length on the x-axis"
            )
        if np.abs(self._length_y - other._length_y) > 1e-8:
            raise ValueError(
                "The two canvases need to have the same length on the y-axis"
            )
        if self._npoints_x != other._npoints_x:
            raise ValueError(
                "The two canvases need to have the same number of points on the x-axis"
            )
        if self._npoints_y != other._npoints_y:
            raise ValueError(
                "The two canvases need to have the same number of points on the y-axis"
            )

        res = DensityCanvas(
            self._origin,
            self._length_x,
            self._length_y,
            self._npoints_x,
            self._npoints_y,
        )
        res._density = self._density + other._density
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __sub__(self, other: Self) -> Self:
        if np.linalg.norm(self._origin - other._origin) > 1e-8:
            raise ValueError("The two canvases need to have the same origin")
        if np.abs(self._length_x - other._length_x) > 1e-8:
            raise ValueError(
                "The two canvases need to have the same length on the x-axis"
            )
        if np.abs(self._length_y - other._length_y) > 1e-8:
            raise ValueError(
                "The two canvases need to have the same length on the y-axis"
            )
        if self._npoints_x != other._npoints_x:
            raise ValueError(
                "The two canvases need to have the same number of points on the x-axis"
            )
        if self._npoints_y != other._npoints_y:
            raise ValueError(
                "The two canvases need to have the same number of points on the y-axis"
            )

        res = DensityCanvas(
            self._origin,
            self._length_x,
            self._length_y,
            self._npoints_x,
            self._npoints_y,
        )
        res._density = self._density - other._density
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __mul__(self, other: Union[Self, numbers.Number]) -> Self:
        if isinstance(other, numbers.Number):
            res = DensityCanvas(
                self._origin,
                self._length_x,
                self._length_y,
                self._npoints_x,
                self._npoints_y,
            )
            if self._empty:
                return res
            res._density = self._density * other
            res._empty = False
            res._density_type = "arithmetic"
            return res

        if np.linalg.norm(self._origin - other._origin) > 1e-8:
            raise ValueError("The two canvases need to have the same origin")
        if np.abs(self._length_x - other._length_x) > 1e-8:
            raise ValueError(
                "The two canvases need to have the same length on the x-axis"
            )
        if np.abs(self._length_y - other._length_y) > 1e-8:
            raise ValueError(
                "The two canvases need to have the same length on the y-axis"
            )
        if self._npoints_x != other._npoints_x:
            raise ValueError(
                "The two canvases need to have the same number of points on the x-axis"
            )
        if self._npoints_y != other._npoints_y:
            raise ValueError(
                "The two canvases need to have the same number of points on the y-axis"
            )

        res = DensityCanvas(
            self._origin,
            self._length_x,
            self._length_y,
            self._npoints_x,
            self._npoints_y,
        )
        if self._empty or other._empty:
            return res
        res._density = self._density * other._density
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __rmul__(self, other: Self) -> Self:
        return self * other

    def __pow__(self, other: numbers.Number) -> Self:
        res = DensityCanvas(
            self._origin,
            self._length_x,
            self._length_y,
            self._npoints_x,
            self._npoints_y,
        )
        res._density = np.power(self._density, other)
        res._empty = False
        res._density_type = "arithmetic"
        return res

    def __abs__(self) -> Self:
        res = DensityCanvas(
            self._origin,
            self._length_x,
            self._length_y,
            self._npoints_x,
            self._npoints_y,
        )
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
        return np.sum(self._density) * self._dA

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

    def filter_density(self, filter_settings: Dict):
        slice = self._density
        density_type = self._density_type
        self.clear_density()
        self.clear_pubo()
        slice_filter = filter_settings.pop("filter_function")
        self._density = slice_filter(slice, **filter_settings)
        self._empty = False
        self._density_type = density_type

    def set_density_from_gaussians(
        self, centers: ArrayLike, amplitude: numbers.Number, variance: numbers.Number
    ):
        self.clear_density()
        self.clear_pubo()
        rvs = []
        for mu in centers:
            rvs.append(multivariate_normal(mu, variance))
        for rv in rvs:
            self._density += amplitude * rv.pdf(self._pos)
        self._empty = False
        self._centers = np.array([np.array(c) for c in centers])
        self._variance = variance
        self._amplitude = amplitude
        self._density_type = "gaussians"

    def set_randomized_gaussian_density(
        self,
        n_centers: numbers.Number,
        amplitude: numbers.Number,
        variance: numbers.Number,
        minimal_distance: numbers.Number,
        padding: numbers.Number,
        seed: numbers.Number = None,
    ):
        np.random.seed(seed)
        centers = []
        count = 0
        while (len(centers) < n_centers) and count < 10000:
            x = np.random.rand() * (self._length_x - 2 * padding) + (
                self._origin[0] + padding
            )
            y = np.random.rand() * (self._length_y - 2 * padding) + (
                self._origin[1] + padding
            )
            c = np.array([x, y])
            check = True
            for c1 in centers:
                if np.linalg.norm(c - c1) < minimal_distance:
                    check = False
                    count += 1
                    break
            if check:
                centers.append(c)
        if len(centers) < n_centers:
            raise ValueError(
                "It was not possible to generate that many Gaussians on this canvas"
            )
        self.gaussian_centers = centers # save the random centers in gaussian_centers
        self.set_density_from_gaussians(centers, amplitude, variance)

    def set_lattice(self, lattice: Lattice, centering=True):
        self.clear_lattice()
        self.clear_pubo()
        if self._length_x < lattice._length_x:
            raise ValueError(
                "The lattice does not fit in the canvans along the x direction"
            )
        if self._length_y < lattice._length_y:
            raise ValueError(
                "The lattice does not fit in the canvans along the y direction"
            )
        self._lattice = lattice

        shift = self._center - lattice._center if centering else np.zeros(2)
        self._lattice._coords += shift
        self._lattice._length_x = self._length_x
        self._lattice._length_y = self._length_y
        self._lattice._center = (self._length_x / 2, self._length_y / 2)

        try:
            self._lattice._history = [
                np.array(h) + shift for h in self._lattice._history
            ]
        except AttributeError:
            pass

    def get_lattice(self, minimal_spacing: float = None):
        if not minimal_spacing:
            scale_factor = 1
        else:
            distances = []
            for i in range(len(self._lattice._coords)):
                for j in range(i+1, len(self._lattice._coords)):
                    dij = np.linalg.norm(coords[i]-coords[j])
                    dij_exists = False
                    for d in distances:
                        if abs(d-dij)<1e-8:
                            dij_exists = True
                            break
                    if not dij_exists:
                        distances.append(dij)
            minimal_lattice_distance = min(distances)
            scale_factor = minimal_spacing/minimal_lattice_distance

        return scale_factor*np.array(self._lattice._coords)


    def set_rectangular_lattice(self, num_x, num_y, spacing):
        lattice = Lattice.rectangular(num_x=num_x, num_y=num_y, spacing=spacing)
        self.set_lattice(lattice, centering=True)
    
    def set_triangular_lattice(self, nrows, ncols, spacing):
        lattice = Lattice.triangular(nrows=nrows, ncols=ncols, spacing=spacing)
        self.set_lattice(lattice, centering=True)
    
    def set_hexagonal_lattice(self, nrows, ncols, spacing):
        lattice = Lattice.hexagonal(nrows=nrows, ncols=ncols, spacing=spacing)
        self.set_lattice(lattice, centering=True)

    def set_poisson_disk_lattice(self, spacing: tuple, init_points: ArrayLike = None):
        lattice = Lattice.poisson_disk(
            density=self._density,
            length=(self._length_x, self._length_y),
            spacing=spacing,
            init_points = init_points,
        )
        self.set_lattice(lattice, centering=True)

    def set_canvas_rotation(self, rotation: ArrayLike):
        self.canvas_rotation = rotation

    def clear_lattice(self):
        try:
            del self._lattice
        except AttributeError:
            pass

    def crop_canvas(self, center: Tuple[float], size: Tuple[float]):
        """Crops lattice and density slice by user-specified 2D coordinates."""
        x_inds = (
            int((center[0] - self._origin[0] - size[0] / 2) / self._dx),
            int((center[0] - self._origin[0] + size[0] / 2) / self._dx),
        )
        y_inds = (
            int((center[1] - self._origin[1] - size[1] / 2) / self._dy),
            int((center[1] - self._origin[1] + size[1] / 2) / self._dy),
        )

        cropped_density = self._density[y_inds[0] : y_inds[1], x_inds[0] : x_inds[1]]

        self._origin = np.array(center)-np.array(size)/2
        self._npoints_x = cropped_density.shape[1]
        self._npoints_y = cropped_density.shape[0]
        self._length_x = self._npoints_x * self._dx
        self._length_y = self._npoints_y * self._dy
        self._fill_derived_attributes()
        self.set_density_from_slice(cropped_density)

    def lattice_dynamics(
        self,
        spacing: numbers.Number,
        T: numbers.Number = 1000,
        dt: numbers.Number = 1,
        save_history=False,
        viscosity=0.1,
        centering=True,
    ):
        self._lattice._coords -= (
            self._center - self._lattice._center if centering else np.zeros(2)
        )
        self._lattice.dynamics(
            self._density,
            (self._length_x, self._length_y),
            spacing,
            T,
            dt,
            save_history,
            viscosity,
        )
        self._lattice._coords += (
            self._center - self._lattice._center if centering else np.zeros(2)
        )
        if save_history:
            for h in self._lattice._history:
                for c in h:
                    c += (
                        self._center - self._lattice._center
                        if centering
                        else np.zeros(2)
                    )
        try:
            self.calculate_pubo_coefficients(
                p=self._pubo["p"],
                params=self._pubo["params"],
                high=self._pubo["high"],
                low=self._pubo["low"],
            )
        except AttributeError:
            pass

        
    def calculate_pubo_coefficients(self, p: int, params: ArrayLike, high=None, low=None, efficient_qubo=True):
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
            raise AttributeError(
                "Lattice needs to be defined in order to calculate pubo coefficients"
            )

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
                npoints_y=self._npoints_y,
            )
            stg.set_density_from_gaussians([lattice._coords[i]], *pms)
            return stg

        # calculate using formula
        self._pubo = {

            "coeffs": lpn.Lp_coefficients(lattice._coords, p, _base, _component, params, high, low, efficient_qubo),

            "p": p,
            "params": params,
            "high": high,
            "low": low,
        }

    def get_linear_coefficients(self):
        return np.array(list(self._pubo["coeffs"][1].values()))

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
            raise AttributeError(
                "Linear coefficients need to be calculated before decimation"
            )

        new_coords = [c for i, c in enumerate(lattice._coords) if linear[(i,)] < 0]
        new_lattice = Lattice(
            np.array(new_coords),
            length_x=lattice._length_x,
            length_y=lattice._length_y,
            center=lattice._center,
            lattice_type=f"{lattice._lattice_type}(decimated)",
        )
        pubo = self._pubo
        self.set_lattice(new_lattice, centering=False)
        self.calculate_pubo_coefficients(
            p=pubo["p"],
            params=pubo["params"],
            high=pubo["high"],
            low=pubo["low"],
        )

    def force_lattice_size(self, n: int):
        # check that lattice exists
        try:
            lattice = self._lattice
        except AttributeError:
            raise AttributeError("Lattice needs to be defined in order to decimate it")

        size = min((len(lattice._coords), n))

        # check that coefficients have been calculated
        try:
            linear = self._pubo["coeffs"][1]
        except AttributeError:
            raise AttributeError("Coefficients need to be calculated before decimation")
        except KeyError:
            raise AttributeError(
                "Linear coefficients need to be calculated before decimation"
            )

        # find the linear coefficient of the last point to keep
        threshold_value = sorted(list(linear.values()))[size-1]

        new_coords = [
            c for i, c in enumerate(lattice._coords) if (linear[(i,)]-threshold_value) < 1e-4
        ]
        new_lattice = Lattice(
            np.array(new_coords),
            length_x=lattice._length_x,
            length_y=lattice._length_y,
            center=lattice._center,
            lattice_type=f"{lattice._lattice_type}(forced)",
        )
        pubo = self._pubo
        self.set_lattice(new_lattice, centering=False)
        self.calculate_pubo_coefficients(
            p=pubo["p"],
            params=pubo["params"],
            high=pubo["high"],
            low=pubo["low"],
        )

    def calculate_bitstring_cost_from_coefficients(
        self, bitstring: Union[str, ArrayLike], high=None, low=None
    ) -> numbers.Number:
        try:
            coeffs = self._pubo["coeffs"]
        except AttributeError:
            raise AttributeError(
                "Pubo coefficients need to be computed in order to calculate cost"
            )

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
        for i in range(low, high + 1):
            for idx in coeffs[i]:
                if np.any(bits[list(idx)] == 0):
                    continue
                cost += coeffs[i][idx]
        return float(cost)

    def calculate_bitstring_cost_from_distance(
        self,
        bitstring: Union[str, ArrayLike],
        mixture_params: ArrayLike,
        distance: Callable,
        **kwargs,
    ) -> numbers.Number:
        try:
            lattice = self._lattice
        except AttributeError:
            raise AttributeError(
                "Lattice needs to be defined in order to calculate cost"
            )

        # separate bitstring in list of binaries
        bits = np.array([int(b) for b in bitstring])

        # calculate cost
        candidate_centers = [
            lattice._coords[i] for i in range(len(lattice._coords)) if bits[i] == 1
        ]
        test = DensityCanvas(
            self._origin,
            self._length_x,
            self._length_y,
            self._npoints_x,
            self._npoints_y,
        )
        test.set_density_from_gaussians(candidate_centers, *mixture_params)
        return distance(self, test, **kwargs)

    def calculate_detunings(self, minimal_spacing=5, C6=5420158.53):
        """Calculates the detunings as a function of the linear coefficients.
        The argument is the rydberg interaction coefficient C6 (default one is that
        of rydberg level n=70)"""

        linear = {k: -v for (k,), v in self._pubo["coeffs"][1].items()}
        sum_linear = sum(linear.values())
        weights = {k: v/sum_linear for k,v in linear.items()}
        quadratic = {k: v for k, v in self._pubo["coeffs"][2].items()}
        coords = self.get_lattice(minimal_spacing=minimal_spacing)
        
        # calculate rydberg interaction terms
        rydberg = {}
        for i in range(len(linear)):
            for j in range(i+1, len(linear)):
                dij = np.linalg.norm(coords[i]-coords[j])
                rydberg[(i,j)] = C6/dij**6

        # calculate distances per qubit
        distances = {}
        for i in range(len(linear)):
            all_d = []
            for j in range(len(linear)):
                if i==j:
                    continue
                dij = np.linalg.norm(coords[i]-coords[j])
                all_d.append((j, dij))
            distances[i] = sorted(all_d, key=lambda x: x[1], reverse=True)
        
        # calcualte threshold distances (when sum of interactions win over linear coeff)
        threshold_distances = {}
        for i in linear.keys():
            threshold_distances[i] = 0 #initialize as smallest distance
            if linear[i] < 0:
                threshold_distances[i] = distances[i][0][1] #if negative coeff, set largest distance
            else:
                res = 0
                for j, dij in distances[i]:
                    if i<j:
                        val = quadratic[(i,j)]*weights[j]
                    else:
                        val = quadratic[(j,i)]*weights[j]
                    res += val
                    if res > linear[i]*weights[i]:
                        threshold_distances[i] = dij
                        break

        # calculate sum of interactions
        sum_quadratic = np.zeros(len(linear))
        sum_rydberg = np.zeros(len(linear))
        for idx, i in enumerate(linear.keys()):
            res_q = 0
            num_q = 0
            res_i = 0
            num_i = 0
            for pair, val in quadratic.items():
                if i in pair:
                    d10 = np.linalg.norm(coords[pair[0]]-coords[pair[1]])
                    if d10 < threshold_distances[i]:
                        continue
                    res_q += val
                    num_q += 1
            for pair, val in rydberg.items():
                if i in pair:
                    d10 = np.linalg.norm(coords[pair[0]]-coords[pair[1]])
                    if d10 < threshold_distances[i]:
                        continue
                    res_i += val
                    num_i += 1
            sum_quadratic[idx] = res_q/num_q 
            sum_rydberg[idx] = res_i/num_i
            
        alpha = np.mean(sum_rydberg)/np.mean(sum_quadratic)

        return {k: alpha*v for k,v in linear.items()}

    def plotting_objects(
        self,
        figsize=(10, 8),
        title=None,
        draw_centers=False,
        draw_lattice=False,
        lattice_history=False,
        labels=True,
        draw_connections=False,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_box_aspect(self._length_y / self._length_x)
        c = ax.pcolormesh(self._X, self._Y, self._density, cmap="viridis")
        if draw_lattice:
            try:
                ax.scatter(
                    self._lattice._coords[:, 0],
                    self._lattice._coords[:, 1],
                    color="tab:orange",
                )
                if labels:
                    yshift = self._length_y / 50
                    for i, cd in enumerate(self._lattice._coords):
                        plt.text(cd[0], cd[1] - yshift, str(i))
            except AttributeError:
                print("Lattice has not been defined")
        if lattice_history:
            try:
                for trajectory in self._lattice._history:
                    ax.plot(
                        np.array(trajectory)[:, 0],
                        np.array(trajectory)[:, 1],
                        color="tab:orange",
                    )
            except AttributeError:
                print("no history for this lattice")
        if draw_centers:
            try:
                ax.scatter(
                    self._centers[:, 0],
                    self._centers[:, 1],
                    color="red",
                    marker="x",
                    s=120,
                )
            except AttributeError:
                print("Centers have not been defined")
        if draw_connections:
            try:
                coords = self._lattice._coords
                n = len(coords)
                ref = np.min(np.abs(list(self._pubo["coeffs"][1].values())))
                interaction = self._pubo["coeffs"][2]
                for i in range(n):
                    for j in range(i + 1, n):
                        if interaction[(i, j)] > ref:
                            ax.plot(
                                [coords[i][0], coords[j][0]],
                                [coords[i][1], coords[j][1]],
                                color="k",
                                alpha=0.2,
                            )
            except AttributeError:
                print("Can't draw connections without PUBO coefficients")

        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        if title:
            ax.set_title(title)
        fig.colorbar(c)
        return fig, ax

    def draw(
        self,
        figsize=(10, 8),
        title=None,
        draw_centers=False,
        draw_lattice=False,
        lattice_history=False,
        labels=True,
        draw_connections=False,
    ):
        fig, ax = self.plotting_objects(
            figsize,
            title,
            draw_centers,
            draw_lattice,
            lattice_history,
            labels,
            draw_connections,
        )
        plt.show()
