extern crate rand;

use ksuccession::{ KSuccession, Color };
use neuralnetwork::{ NeuralNetwork, LayerDescription };
use matrix::Matrix;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use std::iter::Iterator;

pub struct KSuccessionTrainer {
    value_net: NeuralNetwork,
    game_factory: fn () -> KSuccession,
    momentums: Option<Vec<Matrix<f64>>>
}

pub struct Action {
    action: usize
}

impl Action {
    pub fn get_action(&self) -> usize {
        return self.action;
    }
}

pub struct GameTrace {
    winner: Option<Color>,
    actions: Vec<Action>,
    state_values: Vec<Option<f64>>
}

impl GameTrace {
    pub fn get_actions(&self) -> &Vec<Action> {
        return &self.actions;
    }
}

impl KSuccessionTrainer {
    pub fn new(game_factory: fn () -> KSuccession, value_net: NeuralNetwork) -> KSuccessionTrainer {
        return KSuccessionTrainer {
            game_factory: game_factory,
            value_net: value_net,
            momentums: None
        }
    }

    fn player_value(player: Color) -> f64 {
        return match player {
            Color::GREEN => 1_f64,
            Color::RED => -1_f64
        };
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
            match game.game_with_action(action).map(|game| KSuccessionTrainer::to_nn_config(&game)) {
                None => (), // The action is invalid
                Some(nn_config) => {
                    let value = self.value_net.predict(&nn_config)[(0, 0)];
                    // println!("Predicted {} to value {}", game.game_with_action(action).unwrap(), value);

                    match best_action {
                        None => best_action = Some((action, value)),
                        Some((_, prev_best)) => {
                            let player_modifier = KSuccessionTrainer::player_value(game.get_current_player());
                            if value * player_modifier > prev_best {
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

    pub fn selfplay(&self, exploration_rate: f64) -> GameTrace {
        let distr = Uniform::new(0_f64, 1_f64);
        let mut game = (self.game_factory)();

        let mut actions = Vec::with_capacity(game.get_rows() * game.get_columns());
        let mut values = Vec::with_capacity(game.get_rows() * game.get_columns() + 1);
        let mut winner = None;

        values.push(Some(self.value_net.predict(&KSuccessionTrainer::to_nn_config(&game))[(0, 0)]));
        for step in 0..(game.get_rows() * game.get_columns()) {
            let mut best_action = if thread_rng().sample(distr) < exploration_rate {
                // Random exploration move
                self.sample_random_action(&game)
            } else {
                self.get_best_action(&game).map(|(action, value)| (action, Some(value)))
            };

            match best_action {
                None => (), // Will never happen due to step limit
                Some((action, value)) => {
                    values.push(value);
                    actions.push(Action { action: action });
                    winner = game.play(action);
                    if winner != None {
                        break;
                    }
                }
            }
        }

        GameTrace {
            winner: winner,
            actions: actions,
            state_values: values
        }
    }

    /**
     * Train the neural network return the average state error
     */
    pub fn train(&mut self, trace: &GameTrace, discount_factor: f64) -> f64 {
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

        let alpha = 0.0002_f64;
        let beta = 0.90_f64;

        let mut error = 0_f64;
        let mut error_terms = 0;

        let get_state_error = |state_value: Option<f64>, value_net: &NeuralNetwork, expect: &Matrix<f64>| {
            state_value
                .map(|val| Matrix::new(1, 1, &|_,_| val))
                .map(|prediction| value_net.error_from_prediction(expect, &prediction))
                .unwrap_or(0_f64)
        };

        {
            let expect = expected(0);
            self.momentums = Some(self.value_net.train(&KSuccessionTrainer::to_nn_config(&game), &expect, alpha, beta, &self.momentums));
            error += get_state_error(trace.state_values[0], &self.value_net, &expect);
            error_terms += 1;
        }

        let mut i = 1;
        for (action, state_value) in trace.actions.iter().zip(trace.state_values.iter().skip(1)) {
            game.play(action.action);
            if *state_value != None {
                let expect = expected(i.min(total_rounds - 1));

                // println!("Training \n{} to {}", game, expect);
                self.momentums = Some(self.value_net.train(&KSuccessionTrainer::to_nn_config(&game), &expect, alpha, beta, &self.momentums));

                error += get_state_error(*state_value, &self.value_net, &expect);
                error_terms += 1;
            }
            i += 1;
        }

        return error / (error_terms as f64);
    }
}
