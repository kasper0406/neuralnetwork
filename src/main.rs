extern crate num;
extern crate rand;

mod matrix;
use matrix::Matrix;
use rand::distributions::{Normal, Distribution};
use rand::{thread_rng, Rng};

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

fn load_kasper_samples() -> Vec<ImageSample> {
    let mut result = vec![];
    for i in 0..10 {
        let filename = format!("./data/digits/{}_handwritten.raw", i);
        let mut file = File::open(&filename).expect("File not found");

        let mut pixels = Vec::with_capacity(16 * 16);
        let mut pixel_buffer = [0; 3];
        while let Ok(read_bytes) = file.read(&mut pixel_buffer) {
            if (read_bytes == 0) {
                break;
            }
            pixels.push(1_f64 - ((pixel_buffer[0] as f64 + pixel_buffer[1] as f64 + pixel_buffer[2] as f64) / (3 * 255) as f64));
        }

        result.push(ImageSample {
            label: Matrix::new(10, 1, &|row, col| if i == row { 1_f64 } else { 0_f64 }),
            values: Matrix::new(16 * 16, 1, &|row, col| pixels[row])
        });
    }

    return result;
}

fn main() {
    let image_size = 16 * 16;

    let mut file = File::open("./data/semeion.data").expect("Data file not found!");
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Failed reading dataset file!");
    let mut samples: Vec<ImageSample> = content.trim().split("\n")
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

    let print_sample = |sample: &Matrix<f64>| {
        for i in 0..16 {
            let mut values = Vec::with_capacity(16);
            for j in 0..16 {
                values.push(sample[(i * 16 + j, 0)].to_string());
            }
            println!("{}", values.join(""));
        }
    };

    /*
    for i in 0..100 {
        print_sample(&samples[i].values);
        println!("{}", &samples[i].label);
        println!("");
    } */

    let sigmoid = &Sigmoid;
    let layers = vec![
        /*
        LayerDescription {
            num_neurons: 50_usize,
            function: sigmoid
        }, */
        LayerDescription {
            num_neurons: 50_usize,
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

    let compute_avg_error = |network: &NeuralNetwork, samples: &[ImageSample]| {
        let total_error = samples.iter().fold(0_f64, |acc, sample| {
            return acc + network.error(&sample.values, &sample.label);
        });
        return total_error / samples.len() as f64;
    };

    thread_rng().shuffle(&mut samples);
    let training_samples = &samples[0..1000];
    let test_samples = &samples[1000..];

    for round in 0..500 {
        println!("Avg error after {} rounds: {} in-sample, {} out-of-sample",
            round, compute_avg_error(&nn, training_samples), compute_avg_error(&nn, test_samples));

        for _ in 0..1000 {
            let sample = rand::thread_rng().choose(&training_samples).unwrap();
            nn.train(&sample.values, &sample.label);
        }
    }

    println!("Avg error after training: {} in-sample, {} out-of-sample",
            compute_avg_error(&nn, training_samples), compute_avg_error(&nn, test_samples));

    for kasper_sample in &load_kasper_samples() {
        let prediction = nn.predict(&kasper_sample.values);
        print_sample(&kasper_sample.values);
        println!("Label:\n{}", kasper_sample.label.transpose());
        println!("Prediction:\n{}", prediction);
    }
}
