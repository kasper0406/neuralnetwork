use std::ops::{Index, IndexMut, Add, Sub};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Color {
    GREEN, RED
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Color::GREEN => return write!(f, "G"),
            Color::RED => return write!(f, "R")
        }
    }
}

#[derive(Clone)]
pub struct KSuccession {
    k: usize,
    rows: usize,
    columns: usize,
    board: Vec<Option<Color>>,
    current_player: Color
}

impl KSuccession {
    pub fn new(rows: usize, columns: usize, k: usize) -> KSuccession {
        return KSuccession {
            k: k,
            rows: rows,
            columns: columns,
            board: vec![None; rows * columns],
            current_player: Color::GREEN
        }
    }

    fn validate_position(&self, (row, col): (i64, i64)) -> Option<(usize, usize)> {
        if 0 <= row && 0 <= col && row < (self.rows as i64) && col < (self.columns as i64) {
            return Some((row as usize, col as usize));
        } else {
            return None;
        }
    }

    fn get_score(&self, row: usize, col: usize, step: (i64, i64)) -> Option<Color> {
        let query_pos = (row as i64, col as i64);
        if self[query_pos] == None {
            return None;
        }

        let mut count = 1;
        let mut cur_pos = (query_pos.0 + step.0, query_pos.1 + step.1);
        while let Some(_) = self.validate_position(cur_pos) {
            if (self[cur_pos] == self[query_pos] && count < self.k) {
                count += 1;
                cur_pos = (cur_pos.0 + step.0, cur_pos.1 + step.1);
            } else {
                break;
            }
        }
        cur_pos = (query_pos.0 - step.0, query_pos.1 - step.1);
        while let Some(_) = self.validate_position(cur_pos) {
            if (self[cur_pos] == self[query_pos] && count < self.k) {
                count += 1;
                cur_pos = (cur_pos.0 - step.0, cur_pos.1 - step.1);
            } else {
                break;
            }
        }

        if (count >= self.k) {
            return self[query_pos];
        }
        return None;
    }

    pub fn get_winner_at_pos(&self, row: usize, col: usize) -> Option<Color> {
        if let Some(winner) = self.get_score(row, col, (1, 0)) {
            return Some(winner);
        }

        if let Some(winner) = self.get_score(row, col, (0, 1)) {
            return Some(winner);
        }

        if let Some(winner) = self.get_score(row, col, (1, 1)) {
            return Some(winner);
        }
        
        return None;
    }

    pub fn get_winner(&self) -> Option<Color> {
        for row in 0..self.rows {
            for col in 0..self.columns {
                if let Some(winner) = self.get_winner_at_pos(row, col) {
                    return Some(winner);
                }
            }
        }
        return None;
    }

    pub fn is_action_valid(&self, action: usize) -> bool {
        return self[(0, action)] == None;
    }

    /**
     * plays a move. Returns the winner of the game.
     */
    pub fn play(&mut self, action: usize) -> Option<Color> {
        assert!(action <= self.columns, "Play must be within range of game");
        assert!(self.is_action_valid(action), "Column must not be full");

        let mut i = self.rows - 1;
        while self[(i, action)] != None {
            i -= 1;
        }
        self[(i, action)] = Some(self.current_player);
        self.current_player = match self.current_player {
            Color::GREEN => Color::RED,
            Color::RED => Color::GREEN
        };

        return self.get_winner_at_pos(i, action);
    }

    pub fn game_with_action(&self, action: usize) -> Option<KSuccession> {
        if !self.is_action_valid(action) {
            return None;
        }

        let mut new_game = self.clone();
        new_game.play(action);
        return Some(new_game);
    }

    pub fn get_current_player(&self) -> Color {
        return self.current_player;
    }

    pub fn get_board(&self) -> &Vec<Option<Color>> {
        return &self.board;
    }

    pub fn get_rows(&self) -> usize {
        return self.rows;
    }

    pub fn get_columns(&self) -> usize {
        return self.columns;
    }
}

impl fmt::Display for KSuccession {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = write!(f, "");
        for row in 0..self.rows {
            for col in 0..self.columns {
                result = match self[(row, col)] {
                    Some(color) => result.and(write!(f, "{} ", color)),
                    None => result.and(write!(f, "  "))
                }
            }
            result = result.and(write!(f, "\n"));
        }
        return result;
    }
}

impl Index<(usize, usize)> for KSuccession {
    type Output = Option<Color>;

    fn index<'a>(&'a self, index: (usize, usize)) -> &'a Option<Color> {
        return &self.board[self.columns * index.0 + index.1];
    }
}

impl Index<(i64, i64)> for KSuccession {
    type Output = Option<Color>;

    fn index<'a>(&'a self, index: (i64, i64)) -> &'a Option<Color> {
        assert!(index.0 >= 0 && index.1 >= 0, "Indexes must be non-negative!");
        return &self.board[self.columns * (index.0 as usize) + (index.1 as usize)];
    }
}

impl IndexMut<(usize, usize)> for KSuccession {
    fn index_mut<'a>(&'a mut self, index: (usize, usize)) -> &'a mut Option<Color> {
        return &mut self.board[self.columns * index.0 + index.1];
    }
}
