from functional_neural_networks.dense import FunctionalDense
import keras_tuner
import tensorflow as tf

class HyperOptFnn(keras_tuner.HyperModel):
    def __init__(self, input_shape, resolution):
        self.input_shape = input_shape
        self.resolution = resolution
        super().__init__(self)

    def build_input_layer(self, wanted_normalisation):
        input = tf.keras.layers.Input(shape=self.input_shape, name="input")
        normalised = None
        if wanted_normalisation:
            norm_axes = list(range(len(self.input_shape) - 1))
            normalised = tf.keras.layers.LayerNormalization(
                axis=norm_axes,
                center=False,
                scale=False,
                epsilon=1e-10,
                name="normalization"
            )(input)
        return input, normalised

    def build_hidden_dense_layers(
            self,
            input_layer,
            num_hidden_layers,
            n_bases,
            bases_types,
        ):
        layer_options = []
        for i_layer in range(num_hidden_layers):
            basis_types = bases_types[i_layer]
            dict_layer = {
                "n_neurons": n_bases[i_layer],
                "basis_options": {
                    "n_functions": 6,
                    "resolution": self.resolution,
                    "basis_type": basis_types,
                },
                "activation": "relu",
                "pooling": False
            }
            layer_options.append(dict_layer)

        layer = input_layer
        for i_layer, layer_option in enumerate(layer_options):
            layer = FunctionalDense(
                **layer_option,
                name=f"dense_{i_layer}"
            )(layer)
        return layer

    def build(self, hp):
        wanted_normalisation = hp.Boolean("wanted_normalisation")
        input, normalised = self.build_input_layer(wanted_normalisation)
        hidden_layer_input = input
        if wanted_normalisation:
            hidden_layer_input = normalised
        num_hidden_layers = hp.Int("hidden_layer", min_value=2, max_value=6, step=1)
        n_bases_hidden = [
            hp.Int(f"units_{i_layer}", min_value=5, max_value=15, step=1) for i_layer in range(num_hidden_layers)
        ]
        # To speed up simulations, use Legendre
        #bases_types_hidden = [
        #    hp.Choice(f"basis_type_{i_layer}", ["Fourier", "Legendre"]) for i_layer in range(num_hidden_layers)
        #]
        bases_types_hidden = ["Legendre" for i_layer in range(num_hidden_layers)]

        hidden_dense_layers = self.build_hidden_dense_layers(
            input_layer=hidden_layer_input,
            num_hidden_layers=num_hidden_layers,
            n_bases=n_bases_hidden,
            bases_types=bases_types_hidden,
        )
        n_bases_output = hp.Int("units_output", min_value=1, max_value=15, step=1)
        # To speed up simulations, use Legendre
        #basis_type_output = hp.Choice("basis_type_output", ["Fourier", "Legendre"])
        basis_type_output = "Legendre"
        output_layer_options = {
            "n_neurons": 1,
            "basis_options": {
                "n_functions": n_bases_output,
                "resolution": self.resolution,
                "basis_type": basis_type_output
            },
            "activation": "linear",
            "pooling": True
        }
        output_layer = FunctionalDense(
            **output_layer_options,
            name="output"
        )(hidden_dense_layers)
        model = tf.keras.Model(inputs=input, outputs=output_layer)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer="adam",
        )
        return model

    def fit(
        self,
        hp,
        model,
        X,
        y,
        epochs=None,
        batch_size=None,
        validation_data=None,
        **kwargs
    ):
        if epochs is None:
            epochs = hp.Choice("epochs", [40, 42, 44, 46, 50, 52, 54, 56, 58, 60])
        if batch_size is None:
            batch_size = hp.Choice("batch_size", [8, 16, 32, 64, 128])
        if validation_data:
            return model.fit(
                x=X,
                y=y,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                **kwargs
            )
        else:
            return model.fit(
            x=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
