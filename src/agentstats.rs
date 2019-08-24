use matrix::matrix::Matrix;
use ksuccession::Color;
use std::fmt;
use std::ops::Sub;

#[derive(Clone)]
pub struct AgentStats {
    wins: Matrix<u64>,
    draws: Matrix<u64>,
    losses: Matrix<u64>,
    elos: Vec<i64>,
    total_games: Vec<u64> 
}

impl AgentStats {
    pub fn new(num_agents: usize) -> AgentStats {
        AgentStats {
            wins: Matrix::new(num_agents, num_agents, &|_,_| 0),
            draws: Matrix::new(num_agents, num_agents, &|_,_| 0),
            losses: Matrix::new(num_agents, num_agents, &|_,_| 0),
            elos: vec![1500; num_agents],
            total_games: vec![0; num_agents]
        }
    }

    pub fn add_win(&mut self, green_player: usize, red_player: usize, winner: Option<Color>) {
        let Q_green = 10_f64.powf((self.elos[green_player] as f64) / 400_f64);
        let Q_red = 10_f64.powf((self.elos[red_player] as f64) / 400_f64);

        let expected_green = Q_green / (Q_green + Q_red);
        let expected_red = Q_red / (Q_green + Q_red);

        let actual_score_green = match winner {
            None => 0.5_f64,
            Some(Color::GREEN) => 1_f64,
            Some(Color::RED) => 0_f64
        };
        let actual_score_red = 1_f64 - actual_score_green;

        let K = 32_f64;
        self.elos[green_player] += (K * (actual_score_green - expected_green)) as i64;
        self.elos[red_player] += (K * (actual_score_red - expected_red)) as i64;

        self.total_games[green_player] += 1;
        self.total_games[red_player] += 1;

        match winner {
            None => self.draws[(green_player, red_player)] += 1,
            Some(Color::GREEN) => self.wins[(green_player, red_player)] += 1,
            Some(Color::RED) => self.losses[(green_player, red_player)] += 1
        }
    }
}

impl<'a> Sub for &'a AgentStats {
    type Output = AgentStats;

    fn sub(self, other: &AgentStats) -> AgentStats {
        assert!(self.wins.rows() == other.wins.rows(), "Row count must be the same!");
        assert!(self.wins.columns() == other.wins.columns(), "Column count must be the same!");

        let mut new_elos = Vec::with_capacity(self.elos.len());
        for i in 0 .. self.elos.len() {
            new_elos.push(self.elos[i] - other.elos[i]);
        }
        
        let mut new_total_games = Vec::with_capacity(self.total_games.len());
        for i in 0 .. self.elos.len() {
            new_total_games.push(self.total_games[i] - other.total_games[i]);
        }

        AgentStats {
            wins: &self.wins - &other.wins,
            draws: &self.draws - &other.draws,
            losses: &self.losses - &other.losses,
            elos: new_elos,
            total_games: new_total_games
        }
    }
}

impl fmt::Display for AgentStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = write!(f, "Win matrix:\n");
        for row in 0..self.wins.rows() {
            for col in 0..self.wins.columns() {
                result = result.and(write!(f, "{}/{}/{}\t",
                    self.wins[(row, col)],
                    self.draws[(row, col)],
                    self.losses[(row, col)]));
            }
            result = result.and(write!(f, "\n"));
        }
        result = result.and(write!(f, "\nTotal games: "));
        for i in 0 .. self.total_games.len() {
            result = result.and(write!(f, "{} ", self.total_games[i]));
        }
        result = result.and(write!(f, "\nELOs:\n"));
        for i in 0 .. self.elos.len() {
            result = result.and(write!(f, "{}\n", self.elos[i]));
        }
        result.and(write!(f, "\n"))
    }
}
