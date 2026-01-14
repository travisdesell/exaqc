class DefaultInnovationNumberGenerator:
    """
    This is the basic class to create new innovation numbers. Each call to it
    simply returns the next integer in order.  This should be used when only
    a single process/thread is generating new Gates, to insure all gates have
    unique innovation numbers.
    """

    def __init__(self, starting_number: int = 0):
        """
        Initializes a default innovation number generator with the provided starting
        number.

        Args:
            starting_number: defaults to 0 for a new search, but could potentially
                higher if utilizing a pre-existing evolved genome, e.g. for transfer
                learning or starting from a checkpoint.
        """

        self.current_innovation_number = starting_number

    def __call__(self) -> int:
        """
        Increments the current innovation number and returns a new
        unique one.

        Returns:
            The next innovation number in the sequence.
        """
        self.current_innovation_number += 1
        return self.current_innovation_number


innovation_number_generator = DefaultInnovationNumberGenerator()
