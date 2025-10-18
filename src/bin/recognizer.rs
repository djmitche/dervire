use dervire::{Re, Recognizer};

// Recognizer
fn main() {
    dbg!(Recognizer::new(Re::parse("d((abc)+())c")));
}
