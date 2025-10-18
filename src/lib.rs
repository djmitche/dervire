#![allow(dead_code, unused_variables)]

// Macro definitions must com first!
#[cfg(test)]
mod test_macros;

mod re;
mod recognizer;

pub use re::Re;
pub use recognizer::Recognizer;
