#![feature(test)]

extern crate num;
extern crate rand;
// extern crate futures;
extern crate crossbeam;
extern crate rayon;
extern crate num_cpus;
extern crate libc;

#[macro_use] extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate bincode;

use bincode::{serialize, deserialize};

mod matrix;
use matrix::Matrix;
use rand::distributions::{Normal, Distribution};
use rand::{thread_rng, Rng, RngCore};
use std::ptr;
use std::collections::HashMap;
use std::cell::UnsafeCell;
use std::thread;
use std::rc::Rc;

mod matrixhandle;
use matrixhandle::MatrixHandle;

mod activationfunction;
use activationfunction::{Relu, Sigmoid, TwoPlayerScore};
use activationfunction::ActivationFunction;

mod neuralnetwork;
use neuralnetwork::NeuralNetwork;
use neuralnetwork::LayerDescription;
use neuralnetwork::DropoutType;
use neuralnetwork::ActivationFunctionDescriptor;
use neuralnetwork::Regulizer;

use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::slice;

// use futures::executor::ThreadPool;
// use futures::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use std::f64;

mod ksuccession;
use ksuccession::{ KSuccession, Color };

mod ksuccessiontrainer;
use ksuccessiontrainer::{ KSuccessionTrainer, Agent, TrainableAgent, HumanAgent, NeuralNetworkAgent, GameTrace, GameDescription };

extern crate test;
use test::Bencher;

#[derive(Clone)]
struct BattleStats {
    agent1_index: usize,
    agent2_index: usize,
    trace: GameTrace
}

fn construct_agent(game_description: GameDescription, layers: &[LayerDescription]) -> NeuralNetworkAgent {
    let sample_game = GameDescription::construct_game(game_description);

    let mut nn = NeuralNetwork::new(sample_game.get_rows() * sample_game.get_columns(), layers.to_vec());
    // nn.set_dropout(DropoutType::Weight(0.10));
    nn.set_dropout(DropoutType::None);
    nn.set_regulizer(Some(Regulizer::WeightPeanalizer(0.00003_f32)));

    NeuralNetworkAgent::new(game_description, nn, 0.2)
}

fn construct_deep_agent(game_description: GameDescription) -> NeuralNetworkAgent {
    let layers = vec![
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 1_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        }
    ];

    construct_agent(game_description, &layers)
}

fn construct_wide_agent(game_description: GameDescription) -> NeuralNetworkAgent {
    let layers = vec![
        LayerDescription {
            num_neurons: 300_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 300_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 300_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 300_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 300_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 1_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        }
    ];

    construct_agent(game_description, &layers)
}

fn construct_agents(game_description: GameDescription) -> Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> {
    let layers = vec![
        /*
        LayerDescription {
            num_neurons: 100_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },*/
        LayerDescription {
            num_neurons: 160_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 80_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 50_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        },
        LayerDescription {
            num_neurons: 1_usize,
            function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
        }
    ];

    let num_agents = 4;
    let mut agents: Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> = Vec::with_capacity(num_agents + 2);

    for i in 0..num_agents {
        let layers_to_use = &layers[i..];
        agents.push(UnsafeCell::new(Mutex::new(construct_agent(game_description, layers_to_use))));
    }

    agents.push(UnsafeCell::new(Mutex::new(construct_deep_agent(game_description))));
    agents.push(UnsafeCell::new(Mutex::new(construct_wide_agent(game_description))));

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
        let game_description = GameDescription::FourInARow;
        let twoplayerscore = &TwoPlayerScore;

        let num_agents = 4;
        let mut agents: Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> = Vec::with_capacity(num_agents);
        let layers = vec![
            LayerDescription {
                num_neurons: 50_usize,
                function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
            },
            LayerDescription {
                num_neurons: 1_usize,
                function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
            }
        ];
        for i in 0..num_agents {
            agents.push(UnsafeCell::new(Mutex::new(construct_agent(game_description, &layers))));
        }

        let trainer = KSuccessionTrainer::new(game_description);
        test::black_box(battle_agents(1, &trainer, &agents));
    });
}

fn battle_agents(batches: usize, trainer: &KSuccessionTrainer, agents: &[UnsafeCell<Mutex<NeuralNetworkAgent>>]) {
    let lock_acquire_mutex = Arc::new(Mutex::new(0));

    let mut agent_battle_indexer: Vec<usize> = Vec::with_capacity(agents.len());
    for i in 0..agents.len() {
        agent_battle_indexer.push(i);
    }

    let mut agent_stats = Matrix::new(agents.len(), agents.len(), &|_,_| 0);
    let mut agent_errors = Matrix::new(agents.len(), 1, &|_,_| 0_f64);

    // TODO(knielsen): Figure out a way to save agent state
    let rounds_per_batch = 5;
    let report_interval = 100;
    let snapshot_interval = 5000;
    let mut prev_agent_stats = agent_stats.clone();
    for i in 0..batches {
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

        if i != 0 && i % snapshot_interval == 0 {
            println!("Snapshotting agents at {}", i);
            serialize_agents(&agents);
        }

        let mut battle_threads = Vec::with_capacity(rounds_per_batch * agents.len());
        let mut battle_stats_per_agent = vec![Vec::with_capacity(rounds_per_batch * 2); agents.len()];

        crossbeam::scope(|battle_scope| {
            for round in 0 .. rounds_per_batch {
                thread_rng().shuffle(&mut agent_battle_indexer);

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

                        let trace = match agent2 {
                            None => {
                                let mut agents: Vec<&mut Agent> = vec![&mut *agent1];
                                thread_trainer.battle(&mut agents)
                            },
                            Some(mut actual_agent2) => {
                                let mut agent1_ref = &mut *agent1;
                                let mut agent2_ref = &mut *actual_agent2;
                                let mut agents: Vec<&mut Agent> = vec![agent1_ref, agent2_ref];
                                thread_trainer.battle(&mut agents)
                            }
                        };

                        return BattleStats {
                            agent1_index: agent1_index,
                            agent2_index: agent2_index,
                            trace: trace
                        };
                    }));
                }
            }

            for battle_thread in battle_threads {
                let stats = battle_thread.join().unwrap();
                
                agent_stats[(stats.agent1_index, stats.agent2_index)] += match stats.trace.get_winner() {
                    Some(Color::GREEN) => 1,
                    Some(Color::RED) => -1,
                    None => 0
                };

                battle_stats_per_agent[stats.agent1_index].push((Color::GREEN, stats.trace.clone()));
                battle_stats_per_agent[stats.agent2_index].push((Color::RED, stats.trace));
            }

        });

        crossbeam::scope(|training_scope| {
            let train_agent = |agent: &mut NeuralNetworkAgent, traces: &[(Color, GameTrace)]| {
                agent.train(traces, 0.8)
            };

            let mut training_threads = Vec::with_capacity(agents.len());
            for agent_index in 0 .. agents.len() {
                let agent_mutex = unsafe { &mut *agents[agent_index].get() };
                let battle_stats_ref = &battle_stats_per_agent[agent_index];

                training_threads.push(training_scope.spawn(move || {
                    let mut agent = agent_mutex.lock().unwrap();
                    (agent_index, train_agent(&mut agent, battle_stats_ref))
                }));
            }
            for training_thread in training_threads {
                let (agent_index, error) = training_thread.join().unwrap();
                agent_errors[(agent_index, 0)] += error; 
            }
        });

        MatrixHandle::synchronize(false);
    }
}

fn serialize_agent(filename: &str, agent: &NeuralNetworkAgent) {
    // let data = serde_json::to_string(agent).expect(&format!("Failed to serialize nn-agent: {}", filename));
    let data = bincode::serialize(&agent).unwrap();
    fs::write(filename, &data).expect(&format!("Failed to write nn-agent: {}", filename));
}

fn serialize_agents(agents: &[UnsafeCell<Mutex<NeuralNetworkAgent>>]) {
    unsafe {
        for (i, agent) in agents.iter().enumerate() {
            let agent = &(*(*(agent.get())).lock().unwrap());
            serialize_agent(&format!("agents/agent_{}.bin", i), &agent);
        }
    }
}

fn deserialize_agent(filename: &str) -> NeuralNetworkAgent {
    // let data = fs::read_to_string(filename).expect(&format!("Failed to read nn-agent file: {}", filename));
    // serde_json::from_str(&data).expect(&format!("Failed to deserialize nn-agent: {}", filename))
    let mut data = Vec::new();
    File::open(filename).expect(&format!("Failed to read nn-agent file: {}", filename)).read_to_end(&mut data);
    bincode::deserialize(&data).unwrap()
}

fn initialize_rayon_thread_pool() {
    let num_cpu_cores = num_cpus::get();

    // The application seems to be a bit memory bound as well, so make a bunch of CPU threads
    let num_threads = 4 * num_cpu_cores;

    println!("Available logical CPU cores: {}", num_cpu_cores);
    println!("Using {} threads", num_threads);
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
}

/*
#[bench]
fn matrix_operations_cpu(b: &mut Bencher) {
    b.iter(|| {
        let A = Matrix::new(1000, 1000, &|row, col| (row + 2 * col) as f32);
        let B = Matrix::new(1000, 1000, &|row, col| (2 * row + col) as f32);

        test::black_box(&(&A + &B).entrywise_product(&A) * &A);
    });
} */

#[bench]
fn matrix_operations_gpu(b: &mut Bencher) {
    b.iter(|| {
        let A = Matrix::new(1000, 1000, &|row, col| (row + 2 * col) as f32);
        let B = Matrix::new(1000, 1000, &|row, col| (2 * row + col) as f32);

        let handle_a = MatrixHandle::from_matrix(A);
        let handle_b = MatrixHandle::from_matrix(B);

        test::black_box(&(&handle_a + &handle_b).entrywise_product(&handle_a) * &handle_a);
    });
}

fn main() {

    initialize_rayon_thread_pool();

    let game_description = GameDescription::FourInARow;
    let trainer = KSuccessionTrainer::new(game_description);
    let mut agents = construct_agents(game_description);

    println!("Loading saved agents...");
    // agents.push(UnsafeCell::new(Mutex::new(deserialize_agent("best_agents/test_agent.json"))));

    println!("Battle agents...");
    battle_agents(100, &trainer, &agents);

    /*
    unsafe {
        let mut agent0 = (&mut *agents[0].get()).lock().unwrap();

        agent0.set_exploration_rate(0_f64);
        agent0.set_verbose(true);

        let human_agent = HumanAgent::new();

        loop {
            let trace = trainer.battle(&*agent0, &human_agent);
            agent0.train(&trace, 0.8, Color::GREEN);
        }
    }
    */
}
