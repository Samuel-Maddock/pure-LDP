
class FreqOracleClient:
    def __init__(self, epsilon, d, index_mapper=None):
        """

        Args:
            epsilon: privacy budget
            d: domain size - not all freq oracles need this, so can be None
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        self.epsilon = epsilon
        self.d = d

        if index_mapper is None:
            self.index_mapper = lambda x: x - 1
        else:
            self.index_mapper = index_mapper

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Method to update params of freq oracle client, should be overridden if more options needed.
        Args:
            epsilon: Optional - privacy budget
            d: Optional - domain size
            index_mapper: Optional - function
        """
        self.epsilon = epsilon if epsilon is not None else self.epsilon
        self.d = d if d is not None else self.d
        self.index_mapper = index_mapper if index_mapper is not None else self.index_mapper

    def _perturb(self, data):
        """
        Used internally to peturb raw data, must be implemented by a FreqOracle
        Args:
            data: user's data item
        """
        assert ("Must Implement")

    def privatise(self, data):
        """
        Public facing method to privatise user's data
        Args:
            data: user's data item
        """
        assert ("Must Implement")