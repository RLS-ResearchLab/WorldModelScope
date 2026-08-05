from .base_model import BaseModel


class BaseWorldModel(BaseModel):

    """
    Complete world model.

    Contains:
    - encoder
    - predictor
    """


    def __init__(
        self,
        encoder,
        predictor
    ):

        super().__init__()

        self.encoder = encoder

        self.predictor = predictor



    def encode(self,x):

        return self.encoder(x)



    def predict(self,z):

        return self.predictor(z)



    def forward(self,x):

        """
        Full pipeline:

        image
          |
        encoder
          |
        latent
          |
        predictor
          |
        future latent

        """

        z = self.encode(x)

        future = self.predict(z)


        return future