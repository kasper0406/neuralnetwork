extern crate num;
extern crate rand;

mod matrix;
use matrix::Matrix;
use rand::distributions::{Normal, Distribution};
use rand::Rng;

mod activationfunction;
use activationfunction::Sigmoid;
use activationfunction::ActivationFunction;

mod neuralnetwork;
use neuralnetwork::NeuralNetwork;
use neuralnetwork::LayerDescription;

use std::fs::File;
use std::io::prelude::*;

struct ImageSample {
    values: Matrix<f64>,
    label: Matrix<f64>
}

fn main() {
    let image_size = 16 * 16;

    let mut file = File::open("/Users/kasper/ML/digits/semeion.data").expect("Data file not found!");
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Failed reading dataset file!");
    let samples: Vec<ImageSample> = content.trim().split("\n")
        .map(|sample| {
            let raw_values: Vec<f64> = sample.trim().split(" ").map(|value| {
                return value.parse().unwrap();
            }).collect();

            return ImageSample {
                values: Matrix::new(image_size, 1, &|row, col| raw_values[row]),
                label: Matrix::new(10, 1, &|row, col| raw_values[image_size + row])
            }
        })
        .collect();

    println!("#samples = {}", samples.len());

    let sigmoid = &Sigmoid;
    let layers = vec![
        LayerDescription {
            num_neurons: 30_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 20_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 10_usize,
            function: sigmoid
        }
    ];

    let mut nn = NeuralNetwork::new(image_size, layers.clone());

    let compute_avg_error = |network: &NeuralNetwork| {
        let total_error = samples.iter().fold(0_f64, |acc, sample| {
            return acc + network.error(&sample.values, &sample.label);
        });
        return total_error / samples.len() as f64;
    };

    for round in 0..100 {
        println!("Avg error after {} rounds: {}", round, compute_avg_error(&nn));

        /*
        for _ in 0..10 {
            let sample = rand::thread_rng().choose(&samples).unwrap();
            nn.train(&sample.values, &sample.label);
        } */

        for sample in &samples {
            nn.train(&sample.values, &sample.label);
        }
    }

    println!("Avg error after = {}", compute_avg_error(&nn));
}
