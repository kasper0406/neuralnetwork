extern crate rand;

use neuralnetwork::LayerDescription;
use neuralnetwork::NeuralNetwork;
use neuralnetwork::Regulizer;
use neuralnetwork::DropoutType;
use neuralnetwork::Layer;
use matrix::Matrix;
use matrixhandle::MatrixHandle;
use activationfunction::ActivationFunction;
use activationfunction::{Relu, Sigmoid, TwoPlayerScore};
use std::ops::{AddAssign, Sub, SubAssign, Mul};
use rand::{Rng};
use rand::distributions::{Normal, Distribution};
use std::iter::{Iterator};

const MAX_INPUT_COLUMNS: usize = 1000;

#[derive(Serialize, Deserialize)]
pub struct InplaceNeuralNetwork<MH> where MH: MatrixHandle {
    layers: Vec<Layer<MH>>,
    dropout: DropoutType,
    regulizer: Option<Regulizer>,

    // TODO(knielsen): Figure out a way not to serialize/deserialize these
    // #[serde(skip_serializing, skip_deserializing)]
    _predictions: Vec<MH>,
    // #[serde(skip_serializing, skip_deserializing)]
    _deltas: Vec<MH>,
    // #[serde(skip_serializing, skip_deserializing)]
    _gradients: Vec<MH>,
    // #[serde(skip_serializing, skip_deserializing)]
    _weights: Vec<MH>,
    // #[serde(skip_serializing, skip_deserializing)]
    _momentums: Vec<MH>,

    // #[serde(skip_serializing, skip_deserializing)]
    // TODO(knielsen): Consider not making this a vec to save some space. But probably doesn't matter
    _chain: Vec<MH>
}

impl<MH: MatrixHandle> InplaceNeuralNetwork<MH>
where
    InplaceNeuralNetwork<MH>: NeuralNetwork<MH>,
    for<'a> f32: Mul<&'a MH, Output=MH>,
    for<'a> MH: AddAssign<&'a MH>,
    for<'a> MH: SubAssign<&'a MH>,
    Sigmoid: ActivationFunction<MH>,
    Relu: ActivationFunction<MH>,
    TwoPlayerScore: ActivationFunction<MH>
{
    fn compute_weights_with_dropouts(&mut self) {
        for (i, layer) in self.layers.iter().enumerate() {
            match self.dropout {
                DropoutType::None => MH::copy(&mut self._weights[i], &layer.weights),
                DropoutType::Weight(rate) => {
                    // TODO(knielsen): Avoid one allocation by doing dropout_elements inplace on the copy
                    MH::copy(&mut self._weights[i], &layer.weights.dropout_elements(rate));
                },
                DropoutType::Neuron(rate) => {
                    if i == self.layers.len() - 1 {
                        // Do not drop out result neurons
                        MH::copy(&mut self._weights[i], &layer.weights);
                    } else {
                        // TODO(knielsen): Avoid one allocation by doing dropout_rows inplace on the copy
                        MH::copy(&mut self._weights[i], &layer.weights.dropout_rows(rate));
                    }
                }
            }
        }
    }

    fn get_regulizer_penalty(&self, weights: &MH) -> MH {
        match self.regulizer {
            None => panic!("get_regulizer_penalty should never be called with None value"),
            Some(Regulizer::WeightPeanalizer(lambda)) => lambda * weights
        }
    }

    fn internal_train(&mut self, input: &MH, expected: &MH, alpha: f32, beta: f32) {
        self.compute_weights_with_dropouts();
        self.predict_for_training(input);

        let num_layers = self.layers.len();

        // Setup chain
        {
            let prediction = self._predictions.last().unwrap();
            MH::copy(&mut self._chain[0], &prediction);
            self._chain[0] -= expected;
            self._chain[0].inplace_entrywise_product(&self._deltas.last().unwrap());
        }

        for i in 0 .. self._weights.len() {

            // self._predictions[num_layers - i - 1] are not used for anything after this.
            // Therefore we can go crazy with side effects to do less GPU memory allocation
            self._predictions[num_layers - i - 1].inplace_add_constant_row(1_f32);
            self._predictions[num_layers - i - 1].inplace_transpose();

            MH::multiply(&mut self._gradients[i], &self._chain[i], &mut self._predictions[num_layers - i - 1]);

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

                {
                    let (chain_split_1, chain_split_2) = self._chain.split_at_mut(i + 1);
                    MH::multiply(&mut chain_split_2[0], &self._weights[num_layers - i - 1], &mut chain_split_1[i]);
                }
                self._chain[i + 1].inplace_entrywise_product(&self._deltas[num_layers - 2 - i]);
            }
        }

        let weight_updates = self.layers.iter_mut().zip(self._gradients.iter().rev());
        for (i, (layer, gradient)) in weight_updates.enumerate() {
            self._momentums[i].inplace_scalar_multiply(beta);
            self._momentums[i] += gradient;
            // self._weights[i] are not used anymore after this. We misuse it for operating on the momentums
            MH::copy(&mut self._weights[i], &self._momentums[i]);
            self._weights[i].inplace_scalar_multiply(-alpha);
            layer.weights += &self._weights[i];
        }
    }

    fn predict_for_training(&mut self, input: &MH) {
        MH::copy(&mut self._predictions[0], input);

        let layers_with_dropout = self.layers.iter().zip(self._weights.iter()).enumerate();
        for (i, (layer, dropout_weights)) in layers_with_dropout {
            let layer_function = layer.get_activation_function();

            self._predictions[i].inplace_add_constant_row(1_f32);
            {
                let (pred_split_1, pred_split_2) = self._predictions.split_at_mut(i + 1);
                MH::multiply(&mut pred_split_2[0], dropout_weights, &mut pred_split_1[i]);
            }
            self._predictions[i].inplace_remove_first_row();

            MH::copy(&mut self._deltas[i], &self._predictions[i + 1]);

            layer_function.inline_evaluate(&mut self._predictions[i + 1]);
            layer_function.inline_derivative(&mut self._deltas[i]);
        }
    }
}

impl<MH: MatrixHandle> NeuralNetwork<MH> for InplaceNeuralNetwork<MH>
where
    for<'a> &'a MH: Sub<&'a MH, Output=MH>,
    for<'a> f32: Mul<&'a MH, Output=MH>,
    for<'a> MH: AddAssign<&'a MH>,
    for<'a> MH: SubAssign<&'a MH>,
    Sigmoid: ActivationFunction<MH>,
    Relu: ActivationFunction<MH>,
    TwoPlayerScore: ActivationFunction<MH>
{
    fn new(input_degree: usize, layers: Vec<LayerDescription>) -> InplaceNeuralNetwork<MH> {
        let weight_distribution = Normal::new(0_f64, 1_f64);

        let mut prev_layer_neurons = input_degree;
        let layers: Vec<Layer<MH>> = layers.iter().map(|layer| {
            let network_layer = Layer {
                function_descriptor: layer.function_descriptor,
                weights: MH::from_matrix(
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
        _predictions.push(MH::of_size(layers[0].weights.columns(), MAX_INPUT_COLUMNS));

        let mut _deltas = Vec::with_capacity(layers.len());
        let mut _gradients = Vec::with_capacity(layers.len());
        let mut _weights = Vec::with_capacity(layers.len());
        let mut _momentums = Vec::with_capacity(layers.len());
        let mut _chain = Vec::with_capacity(layers.len());

        for i in 0 .. layers.len() {
            _predictions.push(MH::of_size(layers[i].weights.rows(), MAX_INPUT_COLUMNS));
            _deltas.push(MH::of_size(layers[i].weights.rows(), MAX_INPUT_COLUMNS));
            _gradients.push(MH::of_size(layers[layers.len() - i - 1].weights.rows(),
                                                  layers[layers.len() - i - 1].weights.columns()));
            _weights.push(MH::of_size(layers[i].weights.rows(),
                                                layers[i].weights.columns()));
            _momentums.push(MH::from_matrix(
                Matrix::new(layers[i].weights.rows(), layers[i].weights.columns(), &|_, _| 0_f32)
            ));
            _chain.push(MH::of_size(layers[layers.len() - i - 1].weights.rows(), MAX_INPUT_COLUMNS));
        }

        return InplaceNeuralNetwork {
            layers: layers,
            dropout: DropoutType::None,
            regulizer: None,

            _predictions: _predictions,
            _deltas: _deltas,
            _gradients: _gradients,
            _weights: _weights,
            _momentums: _momentums,
            _chain: _chain
        }
    }

    fn set_dropout(&mut self, dropout: DropoutType) {
        self.dropout = dropout;
    }

    fn set_regulizer(&mut self, regulizer: Option<Regulizer>) {
        self.regulizer = regulizer;
    }

    fn predict(&mut self, input: &MH) -> MH {
        MH::copy(&mut self._predictions[0], input);

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_function = layer.get_activation_function();

            self._predictions[i].inplace_add_constant_row(1_f32);
            {
                let (pred_split_1, pred_split_2) = self._predictions.split_at_mut(i + 1);
                MH::multiply(&mut pred_split_2[0], &layer.weights, &mut pred_split_1[i]);
            }
            self._predictions[i].inplace_remove_first_row();
            
            layer_function.inline_evaluate(&mut self._predictions[i + 1]);
        }
        self._predictions.last().unwrap().clone()
    }

    fn error_from_prediction(&self, expected: &MH, prediction: &MH) -> f32 {
        assert!(expected.rows() == prediction.rows(), "Expected and prediction should have same number of rows");
        assert!(expected.columns() == prediction.columns(), "Expected and prediction should have same number of columns");

        // TODO(knielsen): Consider implementing this on the GPU
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
        if input.columns() > MAX_INPUT_COLUMNS {
            // TODO(knielsen): Split the inputs such that memory on the GPU can be allocated up front
            panic!("Splitting not yet implemented!");
        } else {
            self.internal_train(input, expected, alpha, beta);
        }

        // TODO(knielsen): Implement momentums properly for the inplace algorithm
        return vec![];
    }
}
