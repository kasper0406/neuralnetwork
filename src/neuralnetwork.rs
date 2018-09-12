extern crate rand;

use matrix::Matrix;
use activationfunction::ActivationFunction;
use std::ops::{Add, AddAssign, Mul, Index};
use num::Zero;
use rand::{thread_rng, Rng};
use rand::distributions::{Uniform, Normal, Distribution};
use std::iter::{Iterator, Rev};

pub struct Layer {
    function: &'static (ActivationFunction<Matrix<f64>> + 'static),
    weights: Matrix<f64>
}

pub enum DropoutType {
    Weight(f64),
    Neuron(f64),
    None
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    dropout: DropoutType,
    regulizer: Option<fn (&Matrix<f64>) -> Matrix<f64>>
}

#[derive(Clone, Copy)]
pub struct LayerDescription {
    pub num_neurons: usize,
    pub function: &'static (ActivationFunction<Matrix<f64>> + 'static),
}

impl NeuralNetwork {
    pub fn new(input_degree: usize, layers: Vec<LayerDescription>) -> NeuralNetwork {
        let weight_distribution = Normal::new(0_f64, 1_f64);

        let mut prev_layer_neurons = input_degree;
        let layers = layers.iter().map(|layer| {
            let network_layer = Layer {
                function: layer.function,
                weights: Matrix::new(layer.num_neurons, prev_layer_neurons + 1, &|row, col| {
                    return weight_distribution.sample(&mut rand::thread_rng());
                })
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

    pub fn set_regulizer(&mut self, regulizer: fn (&Matrix<f64>) -> Matrix<f64>) {
        self.regulizer = Some(regulizer);
    }

    pub fn predict(&self, input: &Matrix<f64>) -> Matrix<f64> {
        return self.layers.iter().fold(input.clone(), |acc, layer| {
            return layer.function.evaluate(&(&layer.weights * &acc.add_constant_row(1_f64)));
        });
    }

    pub fn error(&self, input: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        assert!(expected.columns() == 1, "Expected exactly on column in expectation");

        let prediction = self.predict(input);
        return self.error_from_prediction(expected, &prediction);
    }

    pub fn error_from_prediction(&self, expected: &Matrix<f64>, prediction: &Matrix<f64>) -> f64 {
        assert!(prediction.columns() == 1, "Expected exactly on column in prediction");
        assert!(expected.rows() == prediction.rows(), "Expected and prediction should have same length!");

        let mut error = 0_f64;
        for row in 0..expected.rows() {
            error += (expected[(row, 0)] - prediction[(row, 0)]).powi(2);
        }
        return error;
    }

    fn compute_weights_with_dropouts(&self) -> Vec<Matrix<f64>> {
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

    pub fn train(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, alpha: f64, beta: f64, momentums: &Option<Vec<Matrix<f64>>>) -> Vec<Matrix<f64>> {
        let weights_with_dropout = self.compute_weights_with_dropouts();
        let (predictions, deltas) = self.predict_for_training(input, &weights_with_dropout);
        let prediction = predictions.last().unwrap();

        let num_layers = self.layers.len();
        let mut gradients = Vec::with_capacity(num_layers);

        let mut chain = (prediction - &expected).entrywise_product(deltas.last().unwrap());
        for (i, dropout_weights) in weights_with_dropout.iter().rev().enumerate() {
            let mut gradient = &chain * &predictions[num_layers - i - 1].add_constant_row(1_f64).transpose();
            if self.regulizer.is_some() {
                gradient += self.regulizer.unwrap()(&dropout_weights);
            }

            if i < num_layers - 1 {
                chain = (dropout_weights.transpose().remove_first_row() * chain).entrywise_product(&deltas[num_layers - 2 - i]);
            }

            gradients.push(gradient);
        }

        let unwrapped_momentums = momentums.clone().unwrap_or_else(|| {
            let mut ms = Vec::with_capacity(self.layers.len());
            for layer in self.layers.iter().rev() {
                ms.push(Matrix::new(layer.weights.rows(), layer.weights.columns(), &|row, col| 0_f64));
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

    fn predict_for_training(&self, input: &Matrix<f64>, weights_with_dropout: &Vec<Matrix<f64>>) -> (Vec<Matrix<f64>>, Vec<Matrix<f64>>) {
        let mut results = Vec::with_capacity(self.layers.len() + 1);
        results.push(input.clone());

        let mut deltas = Vec::with_capacity(self.layers.len());
        let layers_with_dropout: Vec<_> = self.layers.iter().zip(weights_with_dropout.iter()).collect();
        for (layer, dropout_weights) in layers_with_dropout {
            let last_result_with_bias = results.last().unwrap().add_constant_row(1_f64);
            
            let eval = dropout_weights * &last_result_with_bias;
            results.push(layer.function.evaluate(&eval));
            deltas.push(layer.function.derivative(&eval));
        }

        return (results, deltas);
    }
}
