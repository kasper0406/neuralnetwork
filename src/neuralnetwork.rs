extern crate rand;

use matrix::Matrix;
use matrixhandle::MatrixHandle;
use activationfunction::ActivationFunction;
use activationfunction::{Relu, Sigmoid, TwoPlayerScore};
use std::ops::{Add, AddAssign, Mul, Index};
use num::Zero;
use rand::{thread_rng, Rng};
use rand::distributions::{Uniform, Normal, Distribution};
use std::iter::{Iterator, Rev};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunctionDescriptor {
    Sigmoid,
    Relu,
    TwoPlayerScore
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    function_descriptor: ActivationFunctionDescriptor,
    weights: MatrixHandle
}

#[derive(Serialize, Deserialize)]
pub enum DropoutType {
    Weight(f32),
    Neuron(f32),
    None
}

#[derive(Serialize, Deserialize)]
pub enum Regulizer {
    WeightPeanalizer(f32)
}

const MAX_INPUT_COLUMNS: usize = 1000;

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    dropout: DropoutType,
    regulizer: Option<Regulizer>,

    // TODO(knielsen): Figure out a way not to serialize/deserialize these
    // #[serde(skip_serializing, skip_deserializing)]
    _predictions: Vec<MatrixHandle>,
    // #[serde(skip_serializing, skip_deserializing)]
    _deltas: Vec<MatrixHandle>,
    // #[serde(skip_serializing, skip_deserializing)]
    _gradients: Vec<MatrixHandle>,
    // #[serde(skip_serializing, skip_deserializing)]
    _weights: Vec<MatrixHandle>,
    // #[serde(skip_serializing, skip_deserializing)]
    _momentums: Vec<MatrixHandle>
}

#[derive(Clone, Copy)]
pub struct LayerDescription {
    pub num_neurons: usize,
    pub function_descriptor: ActivationFunctionDescriptor,
}

impl NeuralNetwork {
    pub fn new(input_degree: usize, layers: Vec<LayerDescription>) -> NeuralNetwork {
        let weight_distribution = Normal::new(0_f64, 1_f64);

        let mut prev_layer_neurons = input_degree;
        let layers: Vec<Layer> = layers.iter().map(|layer| {
            let network_layer = Layer {
                function_descriptor: layer.function_descriptor,
                weights: MatrixHandle::from_matrix(
                    Matrix::new(layer.num_neurons, prev_layer_neurons + 1, &|row, col| {
                        weight_distribution.sample(&mut rand::thread_rng()) as f32
                    })
                )
            };
            prev_layer_neurons = layer.num_neurons;

            return network_layer;
        }).collect();

        // Allocate internal scratchpad matrices
        let mut _predictions = Vec::with_capacity(layers.len() + 1);
        _predictions.push(MatrixHandle::of_size(layers[0].weights.columns(), MAX_INPUT_COLUMNS));

        let mut _deltas = Vec::with_capacity(layers.len());
        let mut _gradients = Vec::with_capacity(layers.len());
        let mut _weights = Vec::with_capacity(layers.len());
        let mut _momentums = Vec::with_capacity(layers.len());
        for i in 0 .. layers.len() {
            _predictions.push(MatrixHandle::of_size(layers[i].weights.columns(), MAX_INPUT_COLUMNS));
            _deltas.push(MatrixHandle::of_size(layers[i].weights.columns(), MAX_INPUT_COLUMNS));
            _gradients.push(MatrixHandle::of_size(layers[layers.len() - i - 1].weights.rows(),
                                                  layers[layers.len() - i - 1].weights.columns()));
            _weights.push(MatrixHandle::of_size(layers[i].weights.rows(),
                                                layers[i].weights.columns()));
            _momentums.push(MatrixHandle::from_matrix(
                Matrix::new(layers[i].weights.rows(), layers[i].weights.columns(), &|_, _| 0_f32)
            ));
        }

        return NeuralNetwork {
            layers: layers,
            dropout: DropoutType::None,
            regulizer: None,

            _predictions: _predictions,
            _deltas: _deltas,
            _gradients: _gradients,
            _weights: _weights,
            _momentums: _momentums
        }
    }

    pub fn set_dropout(&mut self, dropout: DropoutType) {
        self.dropout = dropout;
    }

    pub fn set_regulizer(&mut self, regulizer: Option<Regulizer>) {
        self.regulizer = regulizer;
    }

    pub fn predict(&mut self, input: &MatrixHandle) -> MatrixHandle {
        MatrixHandle::copy(&mut self._predictions[0], input);

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_function = self.get_activation_function(layer);

            self._predictions[i].inplace_add_constant_row(1_f32);
            {
                let (pred_split_1, pred_split_2) = self._predictions.split_at_mut(i + 1);
                MatrixHandle::multiply(&mut pred_split_2[0], &layer.weights, &pred_split_1[i]);
            }
            self._predictions[i].inplace_remove_first_row();
            
            layer_function.inline_evaluate(&mut self._predictions[i + 1]);
        }
        self._predictions.last().unwrap().clone()
    }

    pub fn error(&mut self, input: &MatrixHandle, expected: &MatrixHandle) -> f32 {
        let prediction = self.predict(input);
        return self.error_from_prediction(expected, &prediction);
    }

    pub fn error_from_prediction(&self, expected: &MatrixHandle, prediction: &MatrixHandle) -> f32 {
        assert!(expected.rows() == prediction.rows(), "Expected and prediction should have same number of rows");
        assert!(expected.columns() == prediction.columns(), "Expected and prediction should have same number of columns");

        // TODO(knielsen): Consider implementing this on the GPU
        let errors = MatrixHandle::to_matrix(&(expected - prediction));
        let mut error = 0_f64;
        for col in 0 .. expected.columns() {
            for row in 0 .. expected.rows() {
                error += errors[(row, col)].powi(2) as f64;
            }
        }
        return (error / expected.columns() as f64) as f32;
    }

    fn compute_weights_with_dropouts(&mut self) {
        for (i, layer) in self.layers.iter().enumerate() {
            match self.dropout {
                DropoutType::None => MatrixHandle::copy(&mut self._weights[i], &layer.weights),
                DropoutType::Weight(rate) => {
                    // TODO(knielsen): Avoid one allocation by doing dropout_elements inplace on the copy
                    MatrixHandle::copy(&mut self._weights[i], &layer.weights.dropout_elements(rate));
                },
                DropoutType::Neuron(rate) => {
                    if i == self.layers.len() - 1 {
                        // Do not drop out result neurons
                        MatrixHandle::copy(&mut self._weights[i], &layer.weights);
                    } else {
                        // TODO(knielsen): Avoid one allocation by doing dropout_rows inplace on the copy
                        MatrixHandle::copy(&mut self._weights[i], &layer.weights.dropout_rows(rate));
                    }
                }
            }
        }
    }

    fn get_regulizer_penalty(&self, weights: &MatrixHandle) -> MatrixHandle {
        match self.regulizer {
            None => panic!("get_regulizer_penalty should never be called with None value"),
            Some(Regulizer::WeightPeanalizer(lambda)) => lambda * weights
        }
    }

    pub fn train(&mut self, input: &MatrixHandle, expected: &MatrixHandle, alpha: f32, beta: f32) {
        if input.columns() > MAX_INPUT_COLUMNS {
            // TODO(knielsen): Split the inputs such that memory on the GPU can be allocated up front
            panic!("Splitting not yet implemented!");
        } else {
            return self.internal_train(input, expected, alpha, beta);
        }
    }

    fn internal_train(&mut self, input: &MatrixHandle, expected: &MatrixHandle, alpha: f32, beta: f32) {
        self.compute_weights_with_dropouts();
        self.predict_for_training(input);

        let num_layers = self.layers.len();

        let mut chain = {
            let prediction = self._predictions.last().unwrap();
            (prediction - expected).entrywise_product(self._deltas.last().unwrap())
        };

        for i in 0 .. self._weights.len() {

            // self._predictions[num_layers - i - 1] are not used for anything after this.
            // Therefore we can go crazy with side effects to do less GPU memory allocation
            self._predictions[num_layers - i - 1].inplace_add_constant_row(1_f32);
            self._predictions[num_layers - i - 1].inplace_transpose();

            MatrixHandle::multiply(&mut self._gradients[i], &chain, &self._predictions[num_layers - i - 1]);

            // Clean up the operations
            // TODO(knielsen): Consider not doing the extra transpose. Instead just be nasty and modify the MatrixHandle struct directly
            self._predictions[num_layers - i - 1].inplace_transpose();
            self._predictions[num_layers - i - 1].inplace_remove_first_row();

            if self.regulizer.is_some() {
                self._gradients[i] += self.get_regulizer_penalty(&self._weights[num_layers - i - 1]);
            }

            if i < num_layers - 1 {
                // This is safe as this is a copy of the original weights and dropout weights are not used later
                self._weights[num_layers - i - 1].inplace_transpose();
                self._weights[num_layers - i - 1].inplace_remove_first_row();

                chain = (&self._weights[num_layers - i - 1] * &chain).entrywise_product(&self._deltas[num_layers - 2 - i]);
            }
        }

        let weight_updates = self.layers.iter_mut().zip(self._gradients.iter().rev());
        for (i, (layer, gradient)) in weight_updates.enumerate() {
            self._momentums[i].inplace_scalar_multiply(beta);
            self._momentums[i] += gradient;
            // self._weights[i] are not used anymore after this. We misuse it for operating on the momentums
            MatrixHandle::copy(&mut self._weights[i], &self._momentums[i]);
            self._weights[i].inplace_scalar_multiply(-alpha);
            layer.weights += &self._weights[i];
        }
    }

    fn get_activation_function(&self, layer: &Layer) -> Box<ActivationFunction<MatrixHandle>> {
        match layer.function_descriptor {
            ActivationFunctionDescriptor::Sigmoid => Box::new(Sigmoid {}),
            ActivationFunctionDescriptor::Relu => Box::new(Relu {}),
            ActivationFunctionDescriptor::TwoPlayerScore => Box::new(TwoPlayerScore {})
        }
    }

    fn predict_for_training(&mut self, input: &MatrixHandle) {
        MatrixHandle::copy(&mut self._predictions[0], input);

        let layers_with_dropout = self.layers.iter().zip(self._weights.iter()).enumerate();
        for (i, (layer, dropout_weights)) in layers_with_dropout {
            let layer_function = self.get_activation_function(layer);

            self._predictions[i].inplace_add_constant_row(1_f32);
            {
                let (pred_split_1, pred_split_2) = self._predictions.split_at_mut(i + 1);
                MatrixHandle::multiply(&mut pred_split_2[0], dropout_weights, &pred_split_1[i]);
            }
            self._predictions[i].inplace_remove_first_row();

            MatrixHandle::copy(&mut self._deltas[i], &self._predictions[i + 1]);

            layer_function.inline_evaluate(&mut self._predictions[i + 1]);
            layer_function.inline_derivative(&mut self._deltas[i]);
        }
    }
}
