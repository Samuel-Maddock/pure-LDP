from pure_ldp.core import FreqOracleClient

class ZeroOracleClient(FreqOracleClient):

    def __init__(self):
        """ No arguments

        """
        pass

    def _perturb(self, data):
        """
        Used internally to perturb data.

        Zero oracle does not perturb data and will just return nothing.

        Returns: perturbed data
        """

        return 0

    def privatise(self, data):
        """
        Zero oracle does not privatise data, it just returns 0.

        This oracle is used for experiments (over high dimensional domains),
            it predicts every domain element 0

        Returns: 0
        """

        return 0