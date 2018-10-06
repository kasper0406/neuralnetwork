extern crate rand;

use ksuccession::{ KSuccession, Color };
use neuralnetwork::{ NeuralNetwork, LayerDescription };
use matrix::Matrix;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use std::iter::Iterator;
use std::io;

pub struct KSuccessionTrainer {
    game_factory: fn () -> KSuccession,
}

#[derive(Clone)]
pub struct Action {
    action: usize,
    is_exploratory: bool
}

impl Action {
    pub fn get_action(&self) -> usize {
        return self.action;
    }
}

#[derive(Clone)]
pub struct GameTrace {
    winner: Option<Color>,
    actions: Vec<Action>
}

impl GameTrace {
    pub fn get_actions(&self) -> &Vec<Action> {
        return &self.actions;
    }

    pub fn get_winner(&self) -> &Option<Color> {
        return &self.winner;
    }
}

pub trait Agent {
    fn play(&self, game: &KSuccession) -> Action;
}

pub trait TrainableAgent: Agent {
    fn train(&mut self, trace: &GameTrace, discount_factor: f64) -> f64;
}

pub struct NeuralNetworkAgent {
    value_net: NeuralNetwork,
    momentums: Option<Vec<Matrix<f64>>>,
    game_factory: fn () -> KSuccession,
    exploration_rate: f64,
    verbose: bool
}

impl NeuralNetworkAgent {
    pub fn new(game_factory: fn () -> KSuccession, value_net: NeuralNetwork, exploration_rate: f64) -> NeuralNetworkAgent {
        NeuralNetworkAgent {
            value_net: value_net,
            momentums: None,
            game_factory: game_factory,
            exploration_rate: exploration_rate,
            verbose: false
        }
    }

    pub fn set_exploration_rate(&mut self, exploration_rate: f64) {
        self.exploration_rate = exploration_rate;
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    fn to_nn_config(game: &KSuccession) -> Matrix<f64> {
        return Matrix::new(game.get_rows() * game.get_columns(), 1, &|row, col| {
            return match game.get_board()[row] {
                Some(player) => KSuccessionTrainer::player_value(player),
                None => 0_f64
            }
        });
    }

    fn get_best_action(&self, game: &KSuccession) -> Option<(usize, f64)> {
        let mut best_action = None;
        for action in 0..game.get_columns() {
            match game.game_with_action(action).map(|game| NeuralNetworkAgent::to_nn_config(&game)) {
                None => (), // The action is invalid
                Some(nn_config) => {
                    let value = self.value_net.predict(&nn_config)[(0, 0)];
                    // println!("Predicted {} to value {}", game.game_with_action(action).unwrap(), value);

                    match best_action {
                        None => best_action = Some((action, value)),
                        Some((_, prev_best)) => {
                            let player_modifier = KSuccessionTrainer::player_value(game.get_current_player());
                            if value * player_modifier > prev_best * player_modifier {
                                best_action = Some((action, value));
                            }
                        }
                    }
                }
            }
        }
        return best_action;
    }
    
    fn sample_random_action(&self, game: &KSuccession) -> Option<(usize, Option<f64>)> {
        let mut actions = Vec::with_capacity(game.get_columns());
        for action in 0..game.get_columns() {
            if game.is_action_valid(action) {
                actions.push(action);
            }
        }
        return rand::thread_rng().choose(&actions).map(|val| (*val, None));
    }
}

impl TrainableAgent for NeuralNetworkAgent {
    /**
     * Train the neural network return the average state error
     */
    fn train(&mut self, trace: &GameTrace, discount_factor: f64) -> f64 {
        let mut game = (self.game_factory)();

        let total_rounds: i32 = trace.actions.len() as i32;
        let expected = |round: i32| {
            Matrix::new(1, 1, &|row, col| {
                return match trace.winner {
                    None => 0_f64,
                    Some(player) => KSuccessionTrainer::player_value(player) * discount_factor.powi(total_rounds - 1 - round)
                };
            })
        };

        let alpha = 0.002_f64;
        let beta = 0.90_f64;

        let mut error = 0_f64;
        let mut error_terms = 0;

        {
            let expect = expected(0);
            let game_config = NeuralNetworkAgent::to_nn_config(&game);

            error += self.value_net.error(&game_config, &expect);
            error_terms += 1;

            self.momentums = Some(self.value_net.train(&game_config, &expect, alpha, beta, &self.momentums));
        }

        let mut i = 1;
        for action in &trace.actions {
            game.play(action.action);

            // TODO(knielsen): Condition this check on that the move was made by this agent as well
            //if !action.is_exploratory {
                let expect = expected(i.min(total_rounds - 1));
                let game_config = NeuralNetworkAgent::to_nn_config(&game);

                error += self.value_net.error(&game_config, &expect);
                error_terms += 1;

                // println!("Training \n{} to {}", game, expect);
                self.momentums = Some(self.value_net.train(&game_config, &expect, alpha, beta, &self.momentums));
            //}
            i += 1;
        }

        return error / (error_terms as f64);
    }
}

impl Agent for NeuralNetworkAgent {
    fn play(&self, game: &KSuccession) -> Action {
        let distr = Uniform::new(0_f64, 1_f64);

        let mut best_action = if thread_rng().sample(distr) < self.exploration_rate {
            // Random exploration move
            (self.sample_random_action(&game), true)
        } else {
            (self.get_best_action(&game).map(|(action, value)| (action, Some(value))), false)
        };

        if let (Some((action, value)), is_exploratory) = best_action {
            if self.verbose && value.is_some() {
                println!("Estimated value {}", value.unwrap());
            }
            return Action { action: action, is_exploratory: is_exploratory };
        }

        assert!(false, "This will never happen!");
        Action { action: 0, is_exploratory: false }
    }
}

pub struct HumanAgent { }

impl HumanAgent {
    pub fn new() -> HumanAgent {
        HumanAgent {}
    }
}

impl Agent for HumanAgent {
    fn play(&self, game: &KSuccession) -> Action {
        println!("Input an action for the following game:");
        println!("{}", game);

        let mut action = None;
        while action == None {
            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(_) => {
                    match input.trim().parse::<usize>() {
                        Ok(number) => action = Some(number),
                        Err(e) => println!("Failed to parsed action: {}", e)
                    }
                },
                Err(e) => {
                    println!("Failed to read line: {}", e);
                }
            }
        }

        println!("");
        return Action { action: action.unwrap(), is_exploratory: false };
    }
}

impl KSuccessionTrainer {
    pub fn new(game_factory: fn () -> KSuccession) -> KSuccessionTrainer {
        KSuccessionTrainer {
            game_factory: game_factory
        }
    }

    fn player_value(player: Color) -> f64 {
        return match player {
            Color::GREEN => 1_f64,
            Color::RED => -1_f64
        };
    }

    pub fn battle(&self, agent1: &Agent, agent2: &Agent) -> GameTrace {
        let mut game = (self.game_factory)();
        let mut actions = Vec::with_capacity(game.get_rows() * game.get_columns());
        let mut winner = None;

        let mut current_agent = 0;
        let agents = vec![&agent1, &agent2];

        for _step in 0..(game.get_rows() * game.get_columns()) {
            let action = agents[current_agent].play(&game);
            winner = game.play(action.action);
            actions.push(action);
            if winner != None {
                break;
            }

            current_agent = (current_agent + 1) % agents.len();
        }

        GameTrace {
            winner: winner,
            actions: actions,
        }
    }
}
