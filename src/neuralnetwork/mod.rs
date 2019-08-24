pub mod inplace;
pub mod simple;

use activationfunction::{ Sigmoid, Relu, TwoPlayerScore };
use activationfunction::ActivationFunction;
use matrix::matrixhandle::MatrixHandle;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunctionDescriptor {
    Sigmoid,
    Relu,
    TwoPlayerScore
}

#[derive(Serialize, Deserialize)]
pub struct Layer<MH: MatrixHandle> {
    function_descriptor: ActivationFunctionDescriptor,
    weights: MH
}

impl<MH: MatrixHandle> Layer<MH>
where
    Sigmoid: ActivationFunction<MH>,
    Relu: ActivationFunction<MH>,
    TwoPlayerScore: ActivationFunction<MH>
{
    fn get_activation_function(&self) -> Box<ActivationFunction<MH>>
    {
        match self.function_descriptor {
            ActivationFunctionDescriptor::Sigmoid => Box::new(Sigmoid {}),
            ActivationFunctionDescriptor::Relu => Box::new(Relu {}),
            ActivationFunctionDescriptor::TwoPlayerScore => Box::new(TwoPlayerScore {})
        }
    }
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

#[derive(Clone, Copy)]
pub struct LayerDescription {
    pub num_neurons: usize,
    pub function_descriptor: ActivationFunctionDescriptor,
}

pub trait NeuralNetwork<MH: MatrixHandle> {
    fn new(input_degree: usize, layers: Vec<LayerDescription>) -> Self;

    fn set_dropout(&mut self, dropout: DropoutType);
    fn set_regulizer(&mut self, regulizer: Option<Regulizer>);

    fn predict(&mut self, input: &MH) -> MH;
    fn error_from_prediction(&self, expected: &MH, prediction: &MH) -> f32;

    fn train(&mut self, input: &MH, expected: &MH, alpha: f32, beta: f32, momentums: &Option<Vec<MH>>) -> Vec<MH>;

    fn error(&mut self, input: &MH, expected: &MH) -> f32 {
        let prediction = self.predict(input);
        return self.error_from_prediction(expected, &prediction);
    }
}

