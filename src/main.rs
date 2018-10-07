#![feature(test)]

extern crate num;
extern crate rand;
// extern crate futures;
extern crate crossbeam;
extern crate rayon;

mod matrix;
use matrix::Matrix;
use rand::distributions::{Normal, Distribution};
use rand::{thread_rng, Rng};
use std::ptr;
use std::collections::HashMap;
use std::cell::UnsafeCell;
use std::thread;
use std::rc::Rc;

mod activationfunction;
use activationfunction::{Relu, Sigmoid, TwoPlayerScore};
use activationfunction::ActivationFunction;

mod neuralnetwork;
use neuralnetwork::NeuralNetwork;
use neuralnetwork::LayerDescription;
use neuralnetwork::DropoutType;

use std::fs::File;
use std::io::prelude::*;

// use futures::executor::ThreadPool;
// use futures::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use std::f64;

mod ksuccession;
use ksuccession::{ KSuccession, Color };

mod ksuccessiontrainer;
use ksuccessiontrainer::{ KSuccessionTrainer, Agent, TrainableAgent, HumanAgent, NeuralNetworkAgent, GameTrace };

extern crate test;
use test::Bencher;

struct ImageSample {
    values: Matrix<f64>,
    label: Matrix<f64>
}

struct BattleStats {
    agent1_index: usize,
    agent1_error: Option<f64>,
    agent2_index: usize,
    agent2_error: Option<f64>,
    trace: GameTrace
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

fn construct_agent(game_factory: fn () -> KSuccession, layers: &[LayerDescription]) -> NeuralNetworkAgent {
    let sample_game = game_factory();

    let mut nn = NeuralNetwork::new(sample_game.get_rows() * sample_game.get_columns(), layers.to_vec());
    nn.set_dropout(DropoutType::Weight(0.10));
    nn.set_regulizer(|weights: &Matrix<f64>| {
        return Matrix::new(weights.rows(), weights.columns(), &|row, col| {
            let lambda = 0.00003_f64;
            return lambda * weights[(row, col)];
        });
    });

    NeuralNetworkAgent::new(game_factory, nn, 0.4)
}

fn construct_agents(game_factory: fn () -> KSuccession) -> Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> {
    let twoplayerscore = &TwoPlayerScore;

    let layers = vec![
        LayerDescription {
            num_neurons: 100_usize,
            function: twoplayerscore
        },
        LayerDescription {
            num_neurons: 160_usize,
            function: twoplayerscore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function: twoplayerscore
        },
        LayerDescription {
            num_neurons: 50_usize,
            function: twoplayerscore
        },
        LayerDescription {
            num_neurons: 1_usize,
            function: twoplayerscore
        }
    ];

    let num_agents = 5;
    let mut agents: Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> = Vec::with_capacity(num_agents);

    let sample_game = game_factory();
    for i in 0..num_agents {
        let layers_to_use = &layers[i..];
        agents.push(UnsafeCell::new(Mutex::new(construct_agent(game_factory, layers_to_use))));
    }

    return agents;
}

#[bench]
fn bench_threaded_multiply(b: &mut Bencher) {
    b.iter(|| {
        let a = Matrix::new(200, 200, &|row, col| { row + col });
        let b = Matrix::new(200, 200, &|row, col| { row + col });

        test::black_box(a * b);
    });
}

#[bench]
fn bench_single_thread_multiply(b: &mut Bencher) {
    b.iter(|| {
        let a = Matrix::new(200, 200, &|row, col| { row + col });
        let b = Matrix::new(200, 200, &|row, col| { row + col });

        test::black_box(a.slow_mul(&b));
    });
}

#[bench]
fn bench_entrywise_product_test(b: &mut Bencher) {
    b.iter(|| {
        let a = Matrix::new(1000, 1000, &|row, col| { row + col });
        let b = Matrix::new(1000, 1000, &|row, col| { row - col });
        test::black_box(a.entrywise_product(&b));
    });
}

#[bench]
fn bench_agent_training(b: &mut Bencher) {
    b.iter(|| {
        let game_factory = || KSuccession::new(6, 7, 4);
        let twoplayerscore = &TwoPlayerScore;

        let num_agents = 4;
        let mut agents: Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> = Vec::with_capacity(num_agents);
        let layers = vec![
            LayerDescription {
                num_neurons: 200_usize,
                function: twoplayerscore
            },
            LayerDescription {
                num_neurons: 1_usize,
                function: twoplayerscore
            }
        ];
        for i in 0..num_agents {
            agents.push(UnsafeCell::new(Mutex::new(construct_agent(game_factory, &layers))));
        }

        let trainer = KSuccessionTrainer::new(game_factory);
        test::black_box(battle_agents(1, &trainer, &agents));
    });
}

fn battle_agents(rounds: usize, trainer: &KSuccessionTrainer, agents: &[UnsafeCell<Mutex<NeuralNetworkAgent>>]) {
    let lock_acquire_mutex = Arc::new(Mutex::new(0));

    let mut agent_battle_indexer: Vec<usize> = Vec::with_capacity(agents.len());
    for i in 0..agents.len() {
        agent_battle_indexer.push(i);
    }

    let mut agent_stats = Matrix::new(agents.len(), agents.len(), &|_,_| 0);
    let mut agent_errors = Matrix::new(agents.len(), 1, &|_,_| 0_f64);

    // TODO(knielsen): Figure out a way to save agent state
    let report_interval = 1000;
    let mut prev_agent_stats = agent_stats.clone();
    for i in 0..rounds {
        if i % report_interval == 0 {
            println!("Playing game nr. {}", i);
            for agent_index in 0..agents.len() {
                println!("Agent {} error: {}", agent_index, agent_errors[(agent_index, 0)] / (report_interval as f64));
                agent_errors[(agent_index, 0)] = 0_f64;
            }

            println!("");
            println!("Winner stats (total):");
            println!("{}", &agent_stats);

            println!("Winner stats (this interval)");
            println!("{}", &agent_stats - &prev_agent_stats);
            prev_agent_stats = agent_stats.clone();
        }

        let mut battle_threads = Vec::with_capacity(2 * agents.len());

        thread_rng().shuffle(&mut agent_battle_indexer);

        crossbeam::scope(|battle_scope| {
            for (agent1_index_tmp, agent2_index_tmp) in agent_battle_indexer.iter().zip(0..agents.len()) {
                let agent1_index = (*agent1_index_tmp).clone();
                let agent2_index = agent2_index_tmp.clone();

                let thread_trainer = trainer.clone();

                let mut agent1_mutex;
                let mut agent2_mutex;
                unsafe {
                    agent1_mutex = &mut *agents[agent1_index].get();
                    agent2_mutex = &mut *agents[agent2_index].get();
                }

                let lock_acqure_mutex_clone = lock_acquire_mutex.clone();
                battle_threads.push(battle_scope.spawn(move || {
                    let mut agent1;
                    let mut agent2 = None;
                    {
                        let guard = lock_acqure_mutex_clone.lock().unwrap();
                        agent1 = agent1_mutex.lock().unwrap();
                        if agent1_index != agent2_index {
                            agent2 = Some(agent2_mutex.lock().unwrap());
                        }
                        drop(guard);
                    }

                    let train_agent = |agent: &mut NeuralNetworkAgent, trace: GameTrace| {
                        agent.train(&trace, 0.8)
                    };

                    let mut agent1_error = None;
                    let mut agent2_error = None;
                    let trace;

                    if agent2.is_none() {
                        trace = thread_trainer.battle(&*agent1, &*agent1);
                        agent1_error = Some(train_agent(&mut *agent1, trace.clone()));
                    } else {
                        let mut agent1_ref = &mut *agent1;
                        let mut agent2_ref = &mut *agent2.unwrap();
                        trace = thread_trainer.battle(agent1_ref, agent2_ref);

                        crossbeam::scope(|train_scope| {
                            let trace_clone_1 = trace.clone();
                            let trace_clone_2 = trace.clone();
                            let agent1_trainer_thread = train_scope.spawn(move || {
                                return train_agent(&mut agent1_ref, trace_clone_1)
                            });
                            let agent2_trainer_thread = train_scope.spawn(move || {
                                return train_agent(&mut agent2_ref, trace_clone_2)
                            });

                            agent1_error = Some(agent1_trainer_thread.join().unwrap());
                            agent2_error = Some(agent2_trainer_thread.join().unwrap());
                        });
                    }

                    return BattleStats {
                        agent1_index: agent1_index,
                        agent1_error: agent1_error,
                        agent2_index: agent2_index,
                        agent2_error: agent2_error,
                        trace: trace
                    };
                }));
            }

            for battle_thread in battle_threads {
                let stats = battle_thread.join().unwrap();

                agent_stats[(stats.agent1_index, stats.agent2_index)] += match stats.trace.get_winner() {
                    Some(Color::GREEN) => 1,
                    Some(Color::RED) => -1,
                    None => 0
                };

                agent_errors[(stats.agent1_index, 0)] += stats.agent1_error.unwrap_or(0_f64);
                agent_errors[(stats.agent2_index, 0)] += stats.agent2_error.unwrap_or(0_f64);
            }
        });
    }
}

fn main() {
    let game_factory = || KSuccession::new(6, 7, 4);
    // let game_factory = || KSuccession::new(4, 5, 3);
    let trainer = KSuccessionTrainer::new(game_factory);

    let mut agents = construct_agents(game_factory);
    battle_agents(1000000, &trainer, &agents);

    unsafe {
        let mut agent0 = (&mut *agents[0].get()).lock().unwrap();

        agent0.set_exploration_rate(0_f64);
        agent0.set_verbose(true);

        let human_agent = HumanAgent::new();

        loop {
            let trace = trainer.battle(&*agent0, &human_agent);
            agent0.train(&trace, 0.8);
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

    let sigmoid = &Sigmoid;

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
        for _ in 0..100000 {
            let sample = rand::thread_rng().choose(&training_samples).unwrap();
            momentum = Some(nn.train(&sample.values, &sample.label, 0.02_f64, 0.95_f64, &momentum));
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
