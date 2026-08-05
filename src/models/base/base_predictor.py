from .base_model import BaseModel



class BasePredictor(BaseModel):

    """
    Base class for future prediction modules.
    """

    def __init__(self):
        super().__init__()



    def predict(self, x):
        """
        Predict future latent states.
        """

        raise NotImplementedError(
            "Predictor must implement predict()"
        )



    def forward(self,x):

        return self.predict(x)