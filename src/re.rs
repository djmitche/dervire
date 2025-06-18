use std::{iter::Peekable, rc::Rc};

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Re {
    /// Recognizes nothing.
    Null,
    /// Recognizes an empty string.
    Empty,
    Char(char),
    Concat(Rc<Re>, Rc<Re>),
    Kleene(Rc<Re>),
    Or(Rc<Re>, Rc<Re>),
    And(Rc<Re>, Rc<Re>),
    Neg(Rc<Re>),
}

impl Re {
    pub fn parse(s: impl AsRef<str>) -> Self {
        Self::parse_neg(&mut s.as_ref().chars().peekable())
    }

    fn parse_neg(chars: &mut Peekable<impl Iterator<Item = char>>) -> Re {
        if let Some('!') = chars.peek() {
            chars.next();
            return Re::Neg(Rc::new(Self::parse_neg(chars)));
        };
        Self::parse_kleen(chars)
    }

    fn parse_kleen(chars: &mut Peekable<impl Iterator<Item = char>>) -> Re {
        let mut res = Self::parse_or(chars);
        while let Some(&'*') = chars.peek() {
            chars.next();
            res = Re::Kleene(Rc::new(res));
        }
        res
    }

    fn parse_or(chars: &mut Peekable<impl Iterator<Item = char>>) -> Re {
        let mut res = Self::parse_and(chars);
        while let Some('+') = chars.peek() {
            chars.next();
            let rhs = Self::parse_neg(chars);
            res = Re::Or(Rc::new(res), Rc::new(rhs));
        }
        res
    }

    fn parse_and(chars: &mut Peekable<impl Iterator<Item = char>>) -> Re {
        let mut res = Self::parse_concat(chars);
        while let Some('&') = chars.peek() {
            chars.next();
            let rhs = Self::parse_neg(chars);
            res = Re::And(Rc::new(res), Rc::new(rhs));
        }
        res
    }

    fn parse_concat(chars: &mut Peekable<impl Iterator<Item = char>>) -> Re {
        let mut res = Self::parse_lit(chars);
        while let Some(c) = chars.peek() {
            // Concatenate unless this is some character with other meaning.
            match c {
                '*' | '!' | '+' | '&' => return res,
                _ => (),
            };
            let rhs = Self::parse_neg(chars);
            res = Re::Concat(Rc::new(res), Rc::new(rhs));
        }
        res
    }

    fn parse_lit(chars: &mut Peekable<impl Iterator<Item = char>>) -> Re {
        match chars.next() {
            Some(c) => Re::Char(c),
            None => Re::Empty,
        }
    }

    // TODO: test
    pub fn deriv(&self, c: char) -> Re {
        match self {
            Re::Null => Re::Null,
            Re::Empty => Re::Null,
            Re::Char(c2) if c == *c2 => Re::Empty,
            Re::Char(c2) => Re::Null,
            Re::Concat(lhs, rhs) => {
                if lhs.is_nullable() {
                    Re::Or(Rc::new(lhs.deriv(c)), Rc::new(rhs.deriv(c)))
                } else {
                    lhs.deriv(c)
                }
            }
            Re::Kleene(re) => Re::Concat(Rc::new(re.deriv(c)), re.clone()),
            Re::Or(lhs, rhs) => Re::Or(Rc::new(lhs.deriv(c)), Rc::new(rhs.deriv(c))),
            Re::And(lhs, rhs) => Re::And(Rc::new(lhs.deriv(c)), Rc::new(rhs.deriv(c))),
            Re::Neg(re) => Re::Neg(Rc::new(re.deriv(c))),
        }
    }

    // TODO: test
    fn is_nullable(&self) -> bool {
        match self {
            Re::Null => false,
            Re::Empty => true,
            Re::Char(_) => false,
            Re::Concat(lhs, rhs) => lhs.is_nullable() && rhs.is_nullable(),
            Re::Kleene(_) => true,
            Re::Or(lhs, rhs) => lhs.is_nullable() || rhs.is_nullable(),
            Re::And(lhs, rhs) => lhs.is_nullable() && rhs.is_nullable(),
            Re::Neg(re) => !re.is_nullable(),
        }
    }

    /// Determine if this RE matches the given value, slowly. Returns the number of bytes matched.
    // TODO: incomplete, needs backtracking
    pub fn match_slow(&self, value: &str) -> Option<usize> {
        match self {
            Re::Empty if value.is_empty() => Some(0),
            Re::Char(c) if value.starts_with(*c) => Some(1),
            Re::Concat(lhs, rhs) => {
                let l_offset = lhs.match_slow(value)?;
                let r_offset = rhs.match_slow(&value[l_offset..])?;
                Some(l_offset + r_offset)
            }
            Re::Kleene(re) => {
                let mut offset = 0;
                while let Some(inner_offset) = re.match_slow(&value[offset..]) {
                    offset += inner_offset;
                }
                Some(offset)
            }
            Re::Or(lhs, rhs) => {
                if let Some(offset) = lhs.match_slow(value) {
                    return Some(offset);
                }
                rhs.match_slow(value)
            }
            Re::And(lhs, rhs) => {
                let l_offset = lhs.match_slow(value)?;
                let r_offset = rhs.match_slow(value)?;
                // TODO: Only match when both offsets are the same -- is this correct?
                if l_offset == r_offset {
                    Some(l_offset)
                } else {
                    None
                }
            }
            //Re::Neg(re) => !re.is_nullable(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_empty() {
        assert_eq!(Re::parse(""), Re::Empty);
    }

    #[test]
    fn parse_char() {
        assert_eq!(Re::parse("a"), Re::Char('a'));
    }

    #[test]
    fn parse_neg_char() {
        assert_eq!(Re::parse("!a"), Re::Neg(Rc::new(Re::Char('a'))));
    }

    #[test]
    fn parse_neg_char_kleen() {
        assert_eq!(
            Re::parse("!a*"),
            Re::Neg(Rc::new(Re::Kleene(Rc::new(Re::Char('a')))))
        );
    }

    #[test]
    fn parse_triple_kleen() {
        assert_eq!(
            Re::parse("a***"),
            Re::Kleene(Rc::new(Re::Kleene(Rc::new(Re::Kleene(Rc::new(Re::Char(
                'a'
            )))))))
        );
    }

    #[test]
    fn parse_triple_neg() {
        assert_eq!(
            Re::parse("!!!a"),
            Re::Neg(Rc::new(Re::Neg(Rc::new(Re::Neg(Rc::new(Re::Char('a')))))))
        );
    }

    #[test]
    fn parse_neg_char_concat() {
        assert_eq!(
            Re::parse("!ab"),
            Re::Neg(Rc::new(Re::Concat(
                Rc::new(Re::Char('a')),
                Rc::new(Re::Char('b'))
            )))
        );
    }

    #[test]
    fn parse_or() {
        assert_eq!(
            Re::parse("a+b+c"),
            Re::Or(
                Rc::new(Re::Char('a')),
                Rc::new(Re::Or(Rc::new(Re::Char('b')), Rc::new(Re::Char('c')))),
            )
        );
    }
}
