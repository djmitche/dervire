use crate::re::Re;
use std::rc::Rc;

pub struct Recognizer {
    re: Rc<Re>,
    // TODO: build and use a DFA instead
}

impl Recognizer {
    pub fn new(re: Rc<Re>) -> Self {
        Self { re }
    }

    pub fn matches(&self, input: impl AsRef<str>) -> bool {
        self.re.matches_slow(input)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple_match() {
        let recog = Recognizer::new(Re::parse("ab*c"));
        assert!(recog.matches("ac"));
        assert!(recog.matches("abc"));
        assert!(recog.matches("abbc"));
        assert!(!recog.matches("ababc"));
    }
}
