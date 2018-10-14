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

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    dropout: DropoutType,
    regulizer: Option<Regulizer>
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
        let layers = layers.iter().map(|layer| {
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

        return NeuralNetwork {
            layers: layers,
            dropout: DropoutType::None,
            regulizer: None
        }
    }

    pub fn set_dropout(&mut self, dropout: DropoutType) {
        self.dropout = dropout;
    }

    pub fn set_regulizer(&mut self, regulizer: Option<Regulizer>) {
        self.regulizer = regulizer;
    }

    pub fn predict(&self, input: &MatrixHandle) -> MatrixHandle {
        return self.layers.iter().fold(input.clone(), |acc, layer| {
            let layer_function = self.get_activation_function(layer);
            layer_function.evaluate(&(&layer.weights * &acc.add_constant_row(1_f32)))
        });
    }

    pub fn error(&self, input: &MatrixHandle, expected: &MatrixHandle) -> f32 {
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

    fn compute_weights_with_dropouts(&self) -> Vec<MatrixHandle> {
        return self.layers.iter().enumerate().map(|(i, layer)| {
            match self.dropout {
                DropoutType::None => return layer.weights.clone(),
                DropoutType::Weight(rate) => return layer.weights.dropout_elements(rate),
                DropoutType::Neuron(rate) => {
                    if i == self.layers.len() - 1 {
                        // Do not drop out result neurons
                        return layer.weights.clone();
                    } else {
                        return layer.weights.dropout_rows(rate);
                    }
                }
            }
        }).collect();
    }

    fn get_regulizer_penalty(&self, weights: &MatrixHandle) -> MatrixHandle {
        match self.regulizer {
            None => panic!("get_regulizer_penalty should never be called with None value"),
            Some(Regulizer::WeightPeanalizer(lambda)) => lambda * weights
        }
    }

    pub fn train(&mut self, input: &MatrixHandle, expected: &MatrixHandle, alpha: f32, beta: f32, momentums: &Option<Vec<MatrixHandle>>) -> Vec<MatrixHandle> {
        let weights_with_dropout = self.compute_weights_with_dropouts();
        let (predictions, deltas) = self.predict_for_training(input, &weights_with_dropout);
        let prediction = predictions.last().unwrap();

        let num_layers = self.layers.len();
        let mut gradients = Vec::with_capacity(num_layers);

        let mut chain = (prediction - expected).entrywise_product(deltas.last().unwrap());
        for (i, dropout_weights) in weights_with_dropout.iter().rev().enumerate() {
            let mut gradient = &chain * &predictions[num_layers - i - 1].add_constant_row(1_f32).transpose();
            if self.regulizer.is_some() {
                gradient += self.get_regulizer_penalty(&dropout_weights);
            }

            if i < num_layers - 1 {
                chain = (&dropout_weights.transpose().remove_first_row() * &chain).entrywise_product(&deltas[num_layers - 2 - i]);
            }

            gradients.push(gradient);
        }

        let unwrapped_momentums = momentums.clone().unwrap_or_else(|| {
            let mut ms = Vec::with_capacity(self.layers.len());
            for layer in self.layers.iter().rev() {
                ms.push(MatrixHandle::from_matrix(
                    Matrix::new(layer.weights.rows(), layer.weights.columns(), &|row, col| 0_f32))
                );
            }
            return ms;
        });

        let mut new_momentums = Vec::with_capacity(num_layers);
        let weight_updates = self.layers.iter_mut().rev().zip(gradients.iter().zip(unwrapped_momentums.iter()));
        for (layer, (gradient, momentum)) in weight_updates {
            let new_momentum = &(beta * momentum) + &gradient;
            layer.weights += &new_momentum * -alpha;
            new_momentums.push(new_momentum);
        }

        return new_momentums;
    }

    fn get_activation_function(&self, layer: &Layer) -> Box<ActivationFunction<MatrixHandle>> {
        match layer.function_descriptor {
            ActivationFunctionDescriptor::Sigmoid => Box::new(Sigmoid {}),
            ActivationFunctionDescriptor::Relu => Box::new(Relu {}),
            ActivationFunctionDescriptor::TwoPlayerScore => Box::new(TwoPlayerScore {})
        }
    }

    fn predict_for_training(&self, input: &MatrixHandle, weights_with_dropout: &Vec<MatrixHandle>) -> (Vec<MatrixHandle>, Vec<MatrixHandle>) {
        let mut results = Vec::with_capacity(self.layers.len() + 1);
        results.push(input.clone());

        let mut deltas = Vec::with_capacity(self.layers.len());
        let layers_with_dropout: Vec<_> = self.layers.iter().zip(weights_with_dropout.iter()).collect();
        for (layer, dropout_weights) in layers_with_dropout {
            let layer_function = self.get_activation_function(layer);
            let last_result_with_bias = results.last().unwrap().add_constant_row(1_f32);
            
            let eval = dropout_weights * &last_result_with_bias;
            results.push(layer_function.evaluate(&eval));
            deltas.push(layer_function.derivative(&eval));
        }

        return (results, deltas);
    }
}
