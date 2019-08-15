use neuralnetwork::LayerDescription;
use neuralnetwork::NeuralNetwork;
use neuralnetwork::Regulizer;
use neuralnetwork::DropoutType;
use neuralnetwork::Layer;
use matrix::Matrix;
use matrixhandle::MatrixHandle;
use activationfunction::ActivationFunction;
use activationfunction::{Relu, Sigmoid, TwoPlayerScore};
use rand::distributions::{Normal, Distribution};
use std::iter::{Iterator};
use std::ops::{ Add, Mul, Sub };

#[derive(Serialize, Deserialize)]
pub struct SimpleNeuralNetwork<MH> where MH: MatrixHandle {
    layers: Vec<Layer<MH>>,
    dropout: DropoutType,
    regulizer: Option<Regulizer>
}

impl<MH: MatrixHandle> SimpleNeuralNetwork<MH>
where
    for<'a> f32: Mul<&'a MH, Output=MH>,
    for<'a> &'a MH: Mul<&'a MH, Output=MH>,
    for<'a> &'a MH: Sub<&'a MH, Output=MH>,
    for<'a> &'a MH: Add<&'a MH, Output=MH>,
    Sigmoid: ActivationFunction<MH>,
    Relu: ActivationFunction<MH>,
    TwoPlayerScore: ActivationFunction<MH>
{
    fn compute_weights_with_dropouts(&self) -> Vec<MH> {
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

    fn get_regulizer_penalty(&self, weights: &MH) -> MH {
        match self.regulizer {
            None => MH::from_matrix(Matrix::new(weights.rows(), weights.columns(), &|row, col| 0_f32)),
            Some(Regulizer::WeightPeanalizer(lambda)) => lambda * weights
        }
    }

    fn predict_for_training(&self, input: &MH, weights_with_dropout: &Vec<MH>) -> (Vec<MH>, Vec<MH>) {
        let mut results = Vec::with_capacity(self.layers.len() + 1);
        results.push(input.clone());

        let mut deltas = Vec::with_capacity(self.layers.len());
        let layers_with_dropout: Vec<_> = self.layers.iter().zip(weights_with_dropout.iter()).collect();
        for (layer, dropout_weights) in layers_with_dropout {
            let layer_function = layer.get_activation_function();
            let last_result_with_bias = results.last().unwrap().add_constant_row(1_f32);
            
            let eval = dropout_weights * &last_result_with_bias;
            results.push(layer_function.evaluate(&eval));
            deltas.push(layer_function.derivative(&eval));
        }

        return (results, deltas);
    }
}

impl<MH: MatrixHandle> NeuralNetwork<MH> for SimpleNeuralNetwork<MH>
where
    for<'a> f32: Mul<&'a MH, Output=MH>,
    for<'a> &'a MH: Mul<&'a MH, Output=MH>,
    for<'a> &'a MH: Sub<&'a MH, Output=MH>,
    for<'a> &'a MH: Add<&'a MH, Output=MH>,
    Sigmoid: ActivationFunction<MH>,
    Relu: ActivationFunction<MH>,
    TwoPlayerScore: ActivationFunction<MH>
{
    fn new(input_degree: usize, layers: Vec<LayerDescription>) -> SimpleNeuralNetwork<MH> {
        let weight_distribution = Normal::new(0_f64, 1_f64);

        let mut prev_layer_neurons = input_degree;
        let layers = layers.iter().map(|layer| {
            let network_layer = Layer {
                function_descriptor: layer.function_descriptor,
                weights: MH::from_matrix(Matrix::new(layer.num_neurons, prev_layer_neurons + 1, &|row, col| {
                    return weight_distribution.sample(&mut rand::thread_rng()) as f32;
                }))
            };
            prev_layer_neurons = layer.num_neurons;

            return network_layer;
        }).collect();

        return SimpleNeuralNetwork {
            layers: layers,
            dropout: DropoutType::None,
            regulizer: None
        }
    }

    fn set_dropout(&mut self, dropout: DropoutType) {
        self.dropout = dropout;
    }

    fn set_regulizer(&mut self, regulizer: Option<Regulizer>) {
        self.regulizer = regulizer;
    }

    fn predict(&mut self, input: &MH) -> MH {
        return self.layers.iter().fold(input.clone(), |acc, layer| {
            let layer_function = layer.get_activation_function();
            return layer_function.evaluate(&(&layer.weights * &acc.add_constant_row(1_f32)));
        });
    }

    // TODO(knielsen): Figure out how to re-use this implementation
    fn error_from_prediction(&self, expected: &MH, prediction: &MH) -> f32 {
        assert!(expected.rows() == prediction.rows(), "Expected and prediction should have same number of rows");
        assert!(expected.columns() == prediction.columns(), "Expected and prediction should have same number of columns");

        let errors = MH::to_matrix(&(expected - prediction));
        let mut error = 0_f64;
        for col in 0 .. expected.columns() {
            for row in 0 .. expected.rows() {
                error += errors[(row, col)].powi(2) as f64;
            }
        }
        return (error / expected.columns() as f64) as f32;
    }

    fn train(&mut self, input: &MH, expected: &MH, alpha: f32, beta: f32, momentums: &Option<Vec<MH>>) -> Vec<MH> {
        let weights_with_dropout = self.compute_weights_with_dropouts();
        let (predictions, deltas) = self.predict_for_training(input, &weights_with_dropout);
        let prediction = predictions.last().unwrap();

        let num_layers = self.layers.len();
        let mut gradients = Vec::with_capacity(num_layers);

        let mut chain = (prediction - &expected).entrywise_product(deltas.last().unwrap());
        for (i, dropout_weights) in weights_with_dropout.iter().rev().enumerate() {
            let mut gradient = &chain * &predictions[num_layers - i - 1].add_constant_row(1_f32).transpose();
            // Scale gradient to the number of training points, to make sure the gradient doesn't run crazy
            // TODO(knielsen): Is linear scaling a good idea?
            gradient = (1f32 / (chain.columns() as f32)) * &gradient;

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
                ms.push(MH::from_matrix(
                    Matrix::new(layer.weights.rows(), layer.weights.columns(), &|row, col| 0_f32)));
            }
            return ms;
        });

        let mut new_momentums = Vec::with_capacity(num_layers);
        let weight_updates = self.layers.iter_mut().rev().zip(gradients.iter().zip(unwrapped_momentums.iter()));
        for (layer, (gradient, momentum)) in weight_updates {
            let new_momentum = &(beta * momentum) + &gradient;
            layer.weights += -alpha * &new_momentum;
            new_momentums.push(new_momentum);
        }

        return new_momentums;
    }
}
