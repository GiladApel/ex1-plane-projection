class InvalidInputError(ValueError):
    """Raised for any user-supplied input that cannot be processed (bad file, bad plane, bad CSV)."""


class NotIsoscelesError(InvalidInputError):
    """The plane is valid except that its two legs differ in length; callers may accept it with a warning."""
