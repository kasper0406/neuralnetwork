extern crate num;
extern crate rand;
extern crate futures;

mod matrix;
use matrix::Matrix;
use rand::distributions::{Normal, Distribution};
use rand::{thread_rng, Rng};

mod activationfunction;
use activationfunction::{Relu, Sigmoid};
use activationfunction::ActivationFunction;

mod neuralnetwork;
use neuralnetwork::NeuralNetwork;
use neuralnetwork::LayerDescription;
use neuralnetwork::DropoutType;

use std::fs::File;
use std::io::prelude::*;

use futures::executor::ThreadPool;
use futures::prelude::*;

use std::f64;

mod ksuccession;
use ksuccession::{ KSuccession, Color };

mod ksuccessiontrainer;
use ksuccessiontrainer::{ KSuccessionTrainer };

struct ImageSample {
    values: Matrix<f64>,
    label: Matrix<f64>
}

fn load_kasper_samples() -> Vec<ImageSample> {
    let mut result = vec![];
    for category in &vec![("handwritten", 3), ("machine", 4)] {
        for i in 0..10 {
            let filename = format!("./data/digits/{}_{}.raw", i, category.0);
            let mut file = File::open(&filename).expect("File not found");

            let mut pixels = Vec::with_capacity(16 * 16);
            let mut pixel_buffer = [0; 3];
            while let Ok(read_bytes) = file.read(&mut pixel_buffer) {
                if (read_bytes == 0) {
                    break;
                }
                pixels.push(1_f64 - ((pixel_buffer[0] as f64 + pixel_buffer[1] as f64 + pixel_buffer[2] as f64) / (3 * 255) as f64));

                // Monster hack to adjust for image format
                if (category.1 == 4) {
                    let mut tmp_buffer = [0; 1];
                    file.read(&mut tmp_buffer);
                }
            }

            result.push(ImageSample {
                label: Matrix::new(10, 1, &|row, col| if i == row { 1_f64 } else { 0_f64 }),
                values: Matrix::new(16 * 16, 1, &|row, col| pixels[row])
            });
        }
    }

    return result;
}

fn construct_and_train(alpha: f64, beta: f64, lambda: f64, dropout_rate: f64) -> f64 {
    println!("Do some heavy work!");

    return 1337_f64;
}

fn main() {

    let rows = 6;
    let columns = 7;
    let sigmoid = &Sigmoid;

    let layers = vec![
        LayerDescription {
            num_neurons: 80_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 50_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 25_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 1_usize,
            function: sigmoid
        }
    ];

    let game_factory = || KSuccession::new(6, 7, 4);
    let mut trainer = KSuccessionTrainer::new(
        game_factory,
        NeuralNetwork::new(rows * columns, layers.clone())
    );

    for i in 0..10000 {
        if i % 100 == 0 {
            println!("Playing game {}", i);
        }

        let trace = trainer.selfplay();
        trainer.train(&trace);
    }

    {
        // Let's play a game for fun
        let trace = trainer.selfplay();

        let mut game = game_factory();
        for action in trace.get_actions().iter() {
            game.play(action.get_action());
            println!("{}", game);
        }
    }


    /*
    let actions = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    for action in actions {
        println!("{} playing at column {}", game.get_current_player(), action);
        if let Some(winner) = game.play(action) {
            println!("{}", game);
            println!("{} won the game!", winner);
            break;
        }
        println!("{}", game);
        println!("");
    }
    */

    return;

    // let pool = ThreadPool::new().expect("Failed to create thread pool!");
    // let stream = stream::iter(1..3);

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

    let relu = &Relu;

    let layers = vec![
        /*
        LayerDescription {
            num_neurons: 80_usize,
            function: relu
        }, */
        LayerDescription {
            num_neurons: 50_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 25_usize,
            function: sigmoid
        },
        LayerDescription {
            num_neurons: 10_usize,
            function: sigmoid
        }
    ];

    let mut nn = NeuralNetwork::new(image_size, layers.clone());
    nn.set_dropout(DropoutType::Weight(0.10));
    // nn.set_dropout(DropoutType::Neuron(0.05));

    nn.set_regulizer(|weights: &Matrix<f64>| {
        return Matrix::new(weights.rows(), weights.columns(), &|row, col| {
            let lambda = 0.00003_f64;
            return lambda * weights[(row, col)];
        });
    });

    let compute_avg_error = |network: &NeuralNetwork, samples: &[ImageSample]| {
        let total_error = samples.iter().fold(0_f64, |acc, sample| {
            return acc + network.error(&sample.values, &sample.label);
        });
        return total_error / samples.len() as f64;
    };

    thread_rng().shuffle(&mut samples);
    let training_samples = &samples[0..1000];
    let test_samples = &samples[1000..];

    let mut kasper_samples = load_kasper_samples();
    thread_rng().shuffle(&mut kasper_samples);

    for round in 0..50 {
        let in_sample_error = compute_avg_error(&nn, training_samples);
        println!("Avg error after {} rounds: {} in-sample, {} out-of-sample",
            round, in_sample_error, compute_avg_error(&nn, test_samples));

        let mut momentum = None;
        for _ in 0..1000 {
            let sample = rand::thread_rng().choose(&training_samples).unwrap();
            momentum = Some(nn.train(&sample.values, &sample.label, 0.02_f64, 0.95_f64, momentum));
        }
    }

    println!("Avg error after training: {} in-sample, {} out-of-sample",
            compute_avg_error(&nn, training_samples), compute_avg_error(&nn, test_samples));

    /*
    for kasper_sample in &load_kasper_samples() {
        let prediction = nn.predict(&kasper_sample.values);
        print_sample(&kasper_sample.values);
        println!("Label:\n{}", kasper_sample.label.transpose());
        println!("Prediction:\n{}", prediction);
    }*/

    println!("");
    println!("Classification matrix - rows are labels, columns are predictions:");
    let mut classification_matrix = Matrix::new(10, 10, &|row, col| 0);
    let mut total_misclassified = 0;
    for sample in test_samples {
        let prediction_vector = nn.predict(&sample.values);

        let mut prediction = 0;
        let mut actual = 0;
        for i in 0..10 {
            if prediction_vector[(i, 0)] > prediction_vector[(prediction, 0)] {
                prediction = i;
            }
            if sample.label[(i, 0)] > sample.label[(actual, 0)] {
                actual = i;
            }
        }

        classification_matrix[(actual, prediction)] += 1;
        if actual != prediction {
            total_misclassified += 1;
        }
    }
    println!("{}", classification_matrix);
    println!("Misclassified {} out of {} ({}%)", total_misclassified, test_samples.len(),
        (total_misclassified as f64 / test_samples.len() as f64) * 100_f64);
}
