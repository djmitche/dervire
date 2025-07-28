use crate::re::Re;
use std::{collections::HashMap, rc::Rc};

pub struct Recognizer {
    states: Vec<State>,
}

struct State {
    is_terminal: bool,
    edges: HashMap<char, usize>,
}

impl Recognizer {
    pub fn new(re: Rc<Re>) -> Self {
        let states = Self::build_states(re);
        Self { states }
    }

    pub fn matches(&self, input: impl AsRef<str>) -> bool {
        let mut state = &self.states[0];
        for c in input.as_ref().chars() {
            let Some(edge) = state.edges.get(&c) else {
                return false;
            };
            state = &self.states[*edge];
        }
        state.is_terminal
    }

    fn build_states(re: Rc<Re>) -> Vec<State> {
        #[derive(Default)]
        struct Node {
            is_terminal: bool,
            edges: HashMap<char, Rc<Re>>,
        }
        let mut nodes = Vec::new();
        let mut indices = HashMap::new();
        let mut stack = vec![re];

        while let Some(re) = stack.pop() {
            if indices.contains_key(&re) {
                continue;
            }
            let index = nodes.len();
            nodes.push(Node {
                is_terminal: re.is_nullable(),
                ..Node::default()
            });
            let node = nodes.last_mut().unwrap();
            indices.insert(re.clone(), index);

            // TODO: better way to enumerate character sets
            for c in ['a', 'b', 'c', 'd', 'x', 'y', 'z'] {
                eprintln!("calling {re:?}.deriv({c:?})");
                let deriv = re.deriv(c).canonicalize();
                if deriv.is_null() {
                    continue;
                }
                if !indices.contains_key(&deriv) {
                    stack.push(deriv.clone());
                }
                node.edges.insert(c, deriv);
            }
        }

        nodes
            .into_iter()
            .map(|n| State {
                is_terminal: n.is_terminal,
                edges: n.edges.iter().map(|(c, re)| (*c, indices[re])).collect(),
            })
            .collect()
    }
}

impl std::fmt::Debug for Recognizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Recognizer")?;
        for (i, state) in self.states.iter().enumerate() {
            writeln!(
                f,
                "  {i:?}: {}",
                if state.is_terminal { "(terminal)" } else { "" }
            )?;
            let mut edges: Vec<_> = state.edges.iter().collect();
            edges.sort();
            for (c, i) in edges {
                writeln!(f, "    {c:?} => {i:?}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn match_test(re: &str, input: &str) -> bool {
        let recog = dbg!(Recognizer::new(Re::parse(re)));
        recog.matches(input)
    }

    // Custom match tests for standard regex syntax
    #[test]
    fn test_empty_match() {
        assert!(match_test("", ""));
        assert!(!match_test("", "abc"));
    }

    #[test]
    fn test_char_match() {
        assert!(match_test("a", "a"));
        assert!(!match_test("x", "a"));
        assert!(!match_test("a", ""));
    }

    #[test]
    fn test_concat_match() {
        assert!(match_test("ab", "ab"));
        assert!(!match_test("ab", "abc"));
        assert!(!match_test("abc", "ab"));
        assert!(!match_test("xyz", "abc"));
        assert!(!match_test("abc", "xabc"));
    }

    #[test]
    fn test_kleene_match() {
        assert!(match_test("a*", ""));
        assert!(match_test("a*", "a"));
        assert!(match_test("a*", "aa"));
    }

    #[test]
    fn test_alternation_match() {
        assert!(match_test("a|b", "a"));
        assert!(match_test("a|b", "b"));
        assert!(!match_test("a|b", "c"));
    }

    #[test]
    fn test_double_kleene() {
        assert!(match_test("a*b*", "bbbb"));
        assert!(match_test("a*b*", "abbb"));
        assert!(match_test("a*b*", "aabb"));
        assert!(match_test("a*b*", "aaab"));
        assert!(match_test("a*b*", "aaaa"));
    }

    #[test]
    fn test_kleene_alternation() {
        assert!(match_test("(ab)*", "ababab"));
        assert!(!match_test("(ab)*", "ababa"));
        assert!(match_test("(a*b*)*", "abaaababbb"));
    }
}
