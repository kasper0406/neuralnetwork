extern crate rand;

use matrixhandle::MatrixHandle;
use ksuccession::{ KSuccession, Color };
use neuralnetwork::{ NeuralNetwork, LayerDescription };
use matrix::Matrix;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use std::iter::Iterator;
use std::io;
use std::slice;

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum GameDescription {
    ThreeInARow,
    FourInARow
}

impl GameDescription {
    pub fn construct_game(game_description: GameDescription) -> KSuccession {
        match game_description {
            GameDescription::ThreeInARow => KSuccession::new(4, 5, 3),
            GameDescription::FourInARow => KSuccession::new(6, 7, 4)
        }
    }
}


#[derive(Clone)]
pub struct KSuccessionTrainer {
    game_description: GameDescription,
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
    fn play(&mut self, game: &KSuccession) -> Action;
}

pub trait TrainableAgent: Agent {
    fn train(&mut self, traces: &[(Color, GameTrace)], discount_factor: f64) -> f64;
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetworkAgent {
    value_net: NeuralNetwork,
    game_description: GameDescription,
    exploration_rate: f64,
    verbose: bool
}

impl NeuralNetworkAgent {
    pub fn new(game_description: GameDescription, value_net: NeuralNetwork, exploration_rate: f64) -> NeuralNetworkAgent {
        NeuralNetworkAgent {
            value_net: value_net,
            game_description: game_description,
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

    fn to_nn_config(games: &[KSuccession]) -> MatrixHandle {
        return MatrixHandle::from_matrix(
            Matrix::new(games[0].get_rows() * games[0].get_columns(), games.len(), &|row, col| {
                match games[col].get_board()[row] {
                    Some(player) => KSuccessionTrainer::player_value(player),
                    None => 0_f32
                }
            })
        );
    }

    fn get_best_action(&mut self, game: &KSuccession) -> Option<(usize, f32)> {

        let mut action_numbers = Vec::with_capacity(game.get_columns());
        let mut possible_games = Vec::with_capacity(game.get_columns());
        for action in 0..game.get_columns() {
            match game.game_with_action(action) {
                None => (), // The action is invalid
                Some(possible_game) => {
                    action_numbers.push(action);
                    possible_games.push(possible_game);
                }
            }
        }
        let predictions = MatrixHandle::to_matrix(&self.value_net.predict(
            &NeuralNetworkAgent::to_nn_config(&possible_games)
        ));

        let player_modifier = KSuccessionTrainer::player_value(game.get_current_player());
        let mut best_action = None;

        for i in 0 .. predictions.columns() {
            let value = predictions[(0, i)];
            let action = action_numbers[i];

            match best_action {
                None => best_action = Some((action, value)),
                Some((_, prev_best_value)) => {
                    if value * player_modifier > prev_best_value * player_modifier {
                        best_action = Some((action, value));
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
    fn train(&mut self, traces: &[(Color, GameTrace)], discount_factor: f64) -> f64 {

        let expected = |trace: &GameTrace, round: i32| -> f32 {
            match trace.winner {
                None => 0_f32,
                Some(player) => {
                    let factor = discount_factor.powi(trace.actions.len() as i32 - 1 - round) as f32;
                    KSuccessionTrainer::player_value(player) * factor
                }
            }
        };

        let total_len = traces.iter().map(|(_, trace)| trace.actions.len() + 1).sum();
        let mut games = Vec::with_capacity(total_len);
        let mut expectations = Vec::with_capacity(total_len);

        for (player, trace) in traces {
            let mut game = GameDescription::construct_game(self.game_description);
            let total_rounds: i32 = trace.actions.len() as i32;

            // TODO(knielsen): Figure out how to combine this into one vector
            games.push(game.clone());
            expectations.push(expected(trace, 0));

            for (i, action) in trace.actions.iter().enumerate() {
                game.play(action.action);
                if !action.is_exploratory && game.get_current_player() != *player {
                    let expect = expected(trace, i.min(trace.actions.len() - 1) as i32);
                    games.push(game.clone());
                    expectations.push(expect);
                }
            }
        }

        let game_configs = NeuralNetworkAgent::to_nn_config(&games);
        let expect = MatrixHandle::from_matrix(Matrix::new(1, expectations.len(), &|_, col| expectations[col]));

        let alpha = 0.002_f32;
        let beta = 0.90_f32;
        self.value_net.train(&game_configs, &expect, alpha, beta);

        return self.value_net.error(&game_configs, &expect) as f64;
    }
}

impl Agent for NeuralNetworkAgent {
    fn play(&mut self, game: &KSuccession) -> Action {
        let distr = Uniform::new(0_f64, 1_f64);

        let best_action = if thread_rng().sample(distr) < self.exploration_rate {
            // Random exploration move
            (self.sample_random_action(&game), true)
        } else {
            (self.get_best_action(&game).map(|(action, value)| (action, Some(value as f64))), false)
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
    fn play(&mut self, game: &KSuccession) -> Action {
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
    pub fn new(game_description: GameDescription) -> KSuccessionTrainer {
        KSuccessionTrainer {
            game_description: game_description
        }
    }

    fn player_value(player: Color) -> f32 {
        return match player {
            Color::GREEN => 1_f32,
            Color::RED => -1_f32
        };
    }

    pub fn battle(&self, agents: &mut Vec<&mut Agent>) -> GameTrace {
        let mut game = GameDescription::construct_game(self.game_description);
        let mut actions = Vec::with_capacity(game.get_rows() * game.get_columns());

        let mut winner = None;
        let agents_len = agents.len();

        let mut current_agent_idx = 0;
        for _step in 0..(game.get_rows() * game.get_columns()) {
            let current_agent = &mut agents[current_agent_idx];
            let action = current_agent.play(&game);
            winner = game.play(action.action);
            actions.push(action);
            if winner != None {
                break;
            }

            current_agent_idx = (current_agent_idx + 1) % agents_len;
        }

        GameTrace {
            winner: winner,
            actions: actions,
        }
    }
}
