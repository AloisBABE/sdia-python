import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BoxWindow:
    """une classe utilisée pour représenter des pavés droit en dimension quelconque"""

    def __init__(self, bounds):
        """

        Args:
            bounds (numpy array): list of the box bounds'
        """

        self.bounds = bounds

    def __str__(self):
        r"""BoxWindow: :math:`[a_1, b_1] \times [a_2, b_2] \times \cdots`

        Returns:
            [str]: representation of the box.
        """
        string = "BoxWindow: "
        dim = self.dimension()
        i = 0

        for bound in self.bounds:

            a, b = bound
            if a.dtype == "float64" and a.is_integer():
                a = int(a)
            if b.dtype == "float64" and b.is_integer():
                b = int(b)
            if i == dim - 1:
                string += f"[{a}, {b}]"
            else:
                string += f"[{a}, {b}] x "
            i += 1

        return string

    def __len__(self):
        """returns the integer part of the max length of the box sides'

        Returns:
            [int]: integer part of max length of the box sides'
        """

        return int(np.max(np.diff(self.bounds)))

    def __contains__(self, point):
        """tell if a given point is contained in the box

        Args:
            point (numpy array): list of the coordinates of the point

        Returns:
            [boolean]: True if the point is in the box, False otherwise
        """

        dim = self.dimension()

        assert len(point) == dim

        bounds = self.bounds

        for (a, b), x in zip(self.bounds, point):
            if not a <= x <= b:
                return False

        return True

    def dimension(self):
        """returns the number of dimensions of the box

        Returns:
            [numerical type]: number of dimensions of the box
        """
        return len(self.bounds)

    def volume(self):
        """compute the volume of the box

        Returns:
            [numerical type]: volume of the box
        """
        return np.prod(np.diff(self.bounds))

    def indicator_function(self, points):
        """compute the indicator function of the space delimited by the box at a given set of points

        Args:
            points (numpy array): coordinates of the points

        Returns:
            [numpy array of booleans]: value of the indicator function
        """
        if points.ndim == 2:
            return np.apply_along_axis(lambda x: x in self, 1, points)

        else:
            return np.array([points in self])

    def rand(self, n=1, rng=None):
        """Generate ``n`` points uniformly at random inside the :py:class:`BoxWindow`.

        Args:
            n (int, optional): number of point to generate. Defaults to 1.
            rng ([type], optional): np.random.Generator instance. Defaults to None.

        Returns:
            [numpy list]: list of the generated points
        """

        rng = get_random_number_generator(rng)

        pointArray = rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (n, self.dimension())
        )
        return pointArray

    def center(self):
        """returns the center point of the box

        Returns:
            [numpy list]: center point
        """
        return np.mean(self.bounds, axis=1)


class UnitBoxWindow(BoxWindow):
    def __init__(self, center):
        """subclass of BowWindow for boxes with all sides equal to 1

        Args:
            center (numpy list): list of the coordinates of the central point of the box
        """
        bounds = np.column_stack((center - 0.5, center + 0.5))
        super(UnitBoxWindow, self).__init__(bounds)


class BallWindow:
    """a class used to represents balls in any dimension"""

    def __init__(self, center, radius):
        """

        Args:
            center (numpy list): list of the coordinates of the center of the ball
            radius (numerical type): radius of the ball
        """

        self.center = center
        self.radius = radius

    def __str__(self):
        """ BallWindow: center:[c1 c2 ...] radius:r

        Returns:
            [str]: representation of the ball.
        """
        string = f"BallWindow: center:{self.center} radius:{self.radius}"

        return string

    def __len__(self):
        """returns the diamter of the ball

        Returns:
            [numerical type]: diameter
        """
        return 2 * self.radius

    def __contains__(self, point):
        """tell if a given point is contained in the ball

        Args:
            point (numpy array): list of the coordinates of the point

        Returns:
            [boolean]: True if the point is in the ball, False otherwise
        """
        assert len(point) == self.dimension()

        distance = np.linalg.norm(point - self.center)

        return distance <= self.radius

    def dimension(self):
        """returns the number of dimensions of the ball

        Returns:
            [numerical type]: number of dimensions of the ball
        """
        return len(self.center)

    def volume(self):
        """compute the volume of the box

        Returns:
            [numerical type]: volume of the box
        """
        n = self.dimension()
        R = self.radius
        k, remainder = divmod(n, 2)
        if remainder == 0:
            V = np.pi ** k * R ** n / np.math.factorial(k)
        else:
            V = (
                2
                * np.math.factorial(k)
                * (4 * np.pi) ** k
                * R ** n
                / np.math.factorial(n)
            )
        return V

    def indicator_function(self, point):
        """compute the indicator function of the space delimited by the ball at a given point

        Args:
            point (numpy array): coordinates of the point

        Returns:
            [boolean]: value of the indicator function
        """
        # ? how would you handle multiple points
        if len(np.shape(point)) > 1:
            return np.apply_along_axis(lambda x: x in self, 1, point)
        else:
            return point in self

    def rand(self, n=1, rng=None):
        """Generate ``n`` points uniformly at random inside the :py:class:`BallWindow`.

        Args:
            n (int, optional): number of point to generate. Defaults to 1.
            rng ([type], optional): np.random.Generator instance. Defaults to None.

        Returns:
            [numpy list]: list of the generated points
        """

        dim = self.dimension()
        r = self.radius
        c = self.center

        rng = get_random_number_generator(rng)
        A = rng.normal(0, 1, (dim, n))
        R = rng.uniform(0, 1, n)
        X = R ** (1 / dim) * A / np.linalg.norm(A, axis=0)
        pointArray = np.transpose(X * r) + c

        return pointArray
