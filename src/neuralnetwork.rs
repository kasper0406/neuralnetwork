extern crate rand;

use matrix::Matrix;
use activationfunction::ActivationFunction;
use std::ops::{Add, AddAssign, Mul, Index};
use num::Zero;
use rand::distributions::{Normal, Distribution};
use std::iter::{Iterator, Rev};

pub struct Layer {
    function: &'static (ActivationFunction<Matrix<f64>> + 'static),
    weights: Matrix<f64>
}

pub struct NeuralNetwork {
    layers: Vec<Layer>
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

        return NeuralNetwork { layers: layers }
    }

    pub fn predict(&self, input: &Matrix<f64>) -> Matrix<f64> {
        let mut predictions = self.predict_with_delta(input).0;
        return predictions.pop().unwrap();
    }

    fn predict_with_delta(&self, input: &Matrix<f64>) -> (Vec<Matrix<f64>>, Vec<Matrix<f64>>) {
        let mut results = Vec::with_capacity(self.layers.len() + 1);
        results.push(input.clone());

        let mut deltas = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let last_result_with_bias;
            {
                let last_result = results.last().unwrap();
                last_result_with_bias = Matrix::new(last_result.rows() + 1, last_result.columns(), &|row, column| {
                    if row == 0 {
                        return 1_f64;
                    } else {
                        return last_result[(row - 1, column)];
                    }
                });
            }

            let eval: Matrix<f64> = &(layer.weights) * &last_result_with_bias;
            results.push(layer.function.evaluate(&eval));
            deltas.push(layer.function.derivative(&eval));
        }

        return (results, deltas);
    }

    pub fn error(&self, input: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        assert!(expected.columns() == 1, "Expected exactly on column in expectation");

        let prediction = self.predict(input);
        return self.error_from_prediction(expected, &prediction);
    }

    fn error_from_prediction(&self, expected: &Matrix<f64>, prediction: &Matrix<f64>) -> f64 {
        assert!(prediction.columns() == 1, "Expected exactly on column in prediction");
        assert!(expected.rows() == prediction.rows(), "Expected and prediction should have same length!");

        let mut error = 0_f64;
        for row in 0..expected.rows() {
            error += (expected[(row, 0)] - prediction[(row, 0)]).powi(2);
        }
        return error;
    }

    pub fn train(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>) {
        let (predictions, deltas) = self.predict_with_delta(input);
        let prediction = predictions.last().unwrap();

        // let alpha = 10_f64 * self.error_from_prediction(expected, &prediction);
        let alpha = 1_f64;

        let num_layers = self.layers.len();
        let mut chain = (expected - &prediction).entrywise_product(deltas.last().unwrap());
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let gradient = &chain * &predictions[num_layers - i - 1].add_constant_row(1_f64).transpose();
            if i < num_layers - 1 {
                chain = (layer.weights.transpose().remove_first_row() * chain).entrywise_product(&deltas[num_layers - 2 - i]);
            }

            // Do the weight update.
            // This is safe to do now, as the weights for this layer will not be used again
            let weight_delta = &gradient * alpha;
            layer.weights += weight_delta;
        }
    }
}
