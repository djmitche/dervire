use dervire::{Re, Recognizer};

fn main() {
    dbg!(Recognizer::new(Re::parse("d((abc)+())c")));
}
