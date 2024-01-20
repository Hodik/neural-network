import numpy as np


class LayerLearnData:
    def __init__(self, layer):
        """
        Initializes a new instance of the LayerLearnData class.

        Args:
            layer: The layer object associated with this data.

        Attributes:
            layer: The layer object associated with this data.
            weighted_inputs: An array of weighted inputs for each node in the layer.
            activations: An array of activations for each node in the layer.
            node_values: An array of node values for each node in the layer.
            inputs: A list of inputs received by the layer.
        """
        self.layer = layer
        self.weighted_inputs = np.zeros(layer.n_out)
        self.activations = np.zeros(layer.n_out)
        self.node_values = np.zeros(layer.n_out)
        self.inputs = []

    def __repr__(self) -> str:
        """
        Returns a string representation of the LayerLearnData object.

        Returns:
            A string representation of the LayerLearnData object.
        """
        return f"LayerLearnData(layer={self.layer}, weighted_inputs={self.weighted_inputs}, activations={self.activations}, node_values={self.node_values}, inputs={self.inputs})"


class NetworkLearnData:
    def __init__(self, layers):
        self.layer_data = [LayerLearnData(l) for l in layers]
