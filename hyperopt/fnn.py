from functional_neural_networks.dense import FunctionalDense
import keras_tuner
import tensorflow as tf

class HyperOptFnn(keras_tuner.HyperModel):
    def __init__(self, input_shape, resolution):
        self.input_shape = input_shape
        self.resolution = resolution
        super().__init__(self)

    def build_tuner(
        self,
        max_trials,
        objective="val_loss",
        overwrite=True,
        directory=".",
        project_name="tune_hypermodel",
    ):

        cutom_keras_tuner = keras_tuner.RandomSearch(
            self,
            objective=objective,
            max_trials=max_trials,
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )
        return cutom_keras_tuner


    def build_hidden_layers(self, hp, input_layer):
        num_hidden_layers = hp.Int("hidden_layer", min_value=2, max_value=6, step=1)
        layer_options = []
        for i_layer_hp in range(num_hidden_layers):
            dict_layer = {
                "n_neurons": hp.Int(f"units_{i_layer_hp}", min_value=5, max_value=15, step=1),
                "basis_options": {
                    "n_functions": 6,
                    "resolution": self.resolution,
                    "basis_type": "Legendre",
                },
                "activation": "relu",
                "pooling": False
            }
            layer_options.append(dict_layer)

        layer = input_layer
        for i_layer, layer_option in enumerate(layer_options):
            layer = FunctionalDense(
                **layer_option,
                name=f"FunctionalDense_{i_layer}"
            )(layer)
        return layer

    def build(self, hp):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        #norm_axes = list(range(len(self.input_shape) - 1))
        #norm_layer = tf.keras.layers.LayerNormalization(
        #    axis=norm_axes,
        #    center=False,
        #    scale=False,
        #    epsilon=1e-10,
        #    name="Normalization"
        #)(input_layer)
        #hidden_layers = self.build_hidden_layers(hp=hp, input_layer=norm_layer)
        hidden_layers = self.build_hidden_layers(hp=hp, input_layer=input_layer)
        output_layer_options = {
            "n_neurons": 1,
            "basis_options": {
                "n_functions": 3,
                "resolution": self.resolution,
                "basis_type": "Fourier"
            },
            "activation": "linear",
            "pooling": True
        }
        output_layer = FunctionalDense(
                **output_layer_options,
                name=f"OutputLayer"
            )(hidden_layers)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer="adam",
        )
        return model

    def fit(self, hp, model, X, y, validation_data=None, *args, **kwargs):
        if validation_data:
            return model.fit(
                *args,
                X,
                y,
                validation_data=validation_data,
                epochs=hp.Choice("epochs", [40, 42, 44, 46, 50, 52, 54, 56, 58, 60]),
                **kwargs
            )
        else:
            return model.fit(
                *args,
                X,
                y,
                **kwargs
            )
