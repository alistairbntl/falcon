import math

class ReferenceElement():

    def __init__(self):
        """

        Notes
        -----
        For edge orientation and definitions, see Ervin [201?]
        """
        sqrt_2 = math.sqrt(2)
        self.normal_vectors = [(1./sqrt_2,1./sqrt_2),
                               (-1., 0.),
                               (0., -1.)]
