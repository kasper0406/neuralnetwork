#[macro_use] extern crate serde_derive;

extern crate num;
extern crate rand;

extern crate crossbeam;
extern crate rayon;
extern crate num_cpus;
extern crate libc;

extern crate serde;
extern crate serde_json;
extern crate bincode;

mod matrix;
mod battler;
mod matrixhandle;
mod activationfunction;
mod neuralnetwork;
mod simplematrixhandle;
mod ksuccession;
mod ksuccessiontrainer;
mod agentstats;

fn main() {
    battler::start_battles(100);
}
