use ksuccession::{ KSuccession, Color };
use neuralnetwork::{ NeuralNetwork, LayerDescription };
use matrix::Matrix;

pub struct KSuccessionTrainer {
    value_net: NeuralNetwork,
    game_factory: fn () -> KSuccession
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
    actions: Vec<Action>
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
            value_net: value_net
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
                    let value = self.value_net.predict(&nn_config)[(0, 0)] * KSuccessionTrainer::player_value(game.get_current_player());
                    match best_action {
                        None => best_action = Some((action, value)),
                        Some((_, prev_best)) => {
                            if value > prev_best {
                                best_action = Some((action, value));
                            }
                        }
                    }
                }
            }
        }
        return best_action;
    }

    pub fn selfplay(&self) -> GameTrace {
        let mut game = (self.game_factory)();

        let mut actions = Vec::with_capacity(game.get_rows() * game.get_columns());
        let mut winner = None;

        for step in 0..(game.get_rows() * game.get_columns()) {
            let mut best_action = self.get_best_action(&game);

            match best_action {
                None => (), // Will never happen due to step limit
                Some((action, _)) => {
                    actions.push(Action { action: action });
                    winner = game.play(action);
                    if winner != None {
                        break;
                    }
                }
            }
        }

        return GameTrace {
            winner: winner,
            actions: actions
        }
    }

    pub fn train(&mut self, trace: &GameTrace) {
        let mut game = &(self.game_factory)();

        let expected = &Matrix::new(1, 1, &|row, col| {
            return match trace.winner {
                None => 0_f64,
                Some(player) => KSuccessionTrainer::player_value(player)
            };
        });

        let alpha = 0.02_f64;
        let beta = 0.95_f64;

        self.value_net.train(&KSuccessionTrainer::to_nn_config(game), expected, alpha, beta, None);
        for action in &trace.actions {
            self.value_net.train(&KSuccessionTrainer::to_nn_config(game), expected, alpha, beta, None);
        }
    }
}
