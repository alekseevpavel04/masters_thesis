from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for classification tasks.
    This model consists of fully connected layers with ReLU activation functions.
    """

    def __init__(self, n_feats, n_class, fc_hidden=512):
        """
        Initializes the BaselineModel.

        Args:
            n_feats (int): Number of input features.
            n_class (int): Number of output classes.
            fc_hidden (int): Number of hidden units in the fully connected layers. Default is 512.
        """
        super().__init__()

        # Define the network architecture using Sequential
        self.net = Sequential(
            # First fully connected layer with ReLU activation
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            # Second fully connected layer with ReLU activation
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            # Final fully connected layer to output class scores
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, data_object, **batch):
        """
        Forward pass of the model.

        Args:
            data_object (Tensor): Input tensor containing the features.
            **batch: Additional batch data (unused in this model).

        Returns:
            dict: A dictionary containing the output logits.
        """
        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Returns a string representation of the model, including the total number of parameters
        and the number of trainable parameters.

        Returns:
            str: String representation of the model.
        """
        # Calculate the total number of parameters
        all_parameters = sum(p.numel() for p in self.parameters())
        # Calculate the number of trainable parameters
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Get the default string representation of the model
        result_info = super().__str__()
        # Append the total and trainable parameters to the string
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info