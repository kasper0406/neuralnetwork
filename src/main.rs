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

#[macro_use] extern crate objc;
extern crate objc_foundation;
extern crate cocoa;
extern crate metal;
#[macro_use] extern crate lazy_static;

mod matrix;
mod battler;
mod matrixhandle;
mod activationfunction;
mod neuralnetwork;
mod simplematrixhandle;
mod ksuccession;
mod ksuccessiontrainer;
mod agentstats;
mod digitclassifier;
mod metalmatrixhandle;

// TODO(knielsen): Condition this on Metal feature flag
use cocoa::foundation::NSAutoreleasePool;

fn main() {
    // TODO(knielsen): Condition this on Metal feature flag
    let pool = unsafe { NSAutoreleasePool::new(cocoa::base::nil) };


    // battler::start_battles(10000);
    // digitclassifier::test_digit_classification();

    metalmatrixhandle::test();

    // TODO(knielsen): Condition this on Metal feature flag
    unsafe {
        msg_send![pool, release];
    }
}
