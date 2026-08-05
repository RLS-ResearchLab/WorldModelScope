from .base_model import BaseModel



class BaseEncoder(BaseModel):

    """
    Abstract encoder class.

    Every encoder must convert
    input data into latent features.
    """

    def __init__(self):
        super().__init__()



    def encode(self, x):
        """
        Convert input into representation.

        Must be implemented by child classes.
        """

        raise NotImplementedError(
            "Encoder must implement encode()"
        )



    def forward(self, x):
        """
        PyTorch calls this.
        """

        return self.encode(x)