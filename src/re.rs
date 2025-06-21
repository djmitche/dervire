use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::none_of,
    multi::{fold_many0, many0_count},
    sequence::{delimited, pair, preceded},
};
use std::{cell::OnceCell, rc::Rc};

#[derive(Eq, PartialEq, PartialOrd, Debug, Clone)]
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

thread_local! {
static NULL_SINGLETON: OnceCell<Rc<Re>> = const { OnceCell::new() };
static EMPTY_SINGLETON: OnceCell<Rc<Re>> = const { OnceCell::new() };
}

// Constructors
impl Re {
    pub fn null() -> Rc<Self> {
        NULL_SINGLETON.with(|c| c.get_or_init(|| Rc::new(Re::Null)).clone())
    }
    pub fn empty() -> Rc<Self> {
        EMPTY_SINGLETON.with(|c| c.get_or_init(|| Rc::new(Re::Empty)).clone())
    }
    pub fn char(c: char) -> Rc<Self> {
        Rc::new(Re::Char(c))
    }
    pub fn concat(lhs: Rc<Self>, rhs: Rc<Self>) -> Rc<Self> {
        Rc::new(Re::Concat(lhs, rhs))
    }
    pub fn kleene(re: Rc<Self>) -> Rc<Self> {
        Rc::new(Re::Kleene(re))
    }
    pub fn or(lhs: Rc<Self>, rhs: Rc<Self>) -> Rc<Self> {
        Rc::new(Re::Or(lhs, rhs))
    }
    pub fn and(lhs: Rc<Self>, rhs: Rc<Self>) -> Rc<Self> {
        Rc::new(Re::And(lhs, rhs))
    }
    pub fn neg(re: Rc<Self>) -> Rc<Self> {
        Rc::new(Re::Neg(re))
    }
}

// Recognizers
impl Re {
    pub fn is_null(&self) -> bool {
        *self == Re::Null
    }
    pub fn is_empty(&self) -> bool {
        *self == Re::Empty
    }
    pub fn is_char(&self) -> bool {
        matches!(*self, Re::Char(_))
    }
    pub fn is_concat(&self) -> bool {
        matches!(*self, Re::Concat(_, _))
    }
    pub fn is_kleene(&self) -> bool {
        matches!(*self, Re::Kleene(_))
    }
    pub fn is_or(&self) -> bool {
        matches!(*self, Re::Or(_, _))
    }
    pub fn is_and(&self) -> bool {
        matches!(*self, Re::And(_, _))
    }
    pub fn is_neg(&self) -> bool {
        matches!(*self, Re::Neg(_))
    }
}

// Parsing
impl Re {
    pub fn parse(s: impl AsRef<str>) -> Rc<Self> {
        let input = s.as_ref();
        match Self::parse_concat(input) {
            Ok(("", re)) => re,
            _ => panic!("Invalid regular expression {input:?}"),
        }
    }

    fn parse_concat(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_concat({input:?})");
        fold_many0(Self::parse_neg, Re::empty, |res, subre| {
            if res.is_empty() {
                subre
            } else {
                Re::concat(res, subre)
            }
        })
        .parse(input)
    }

    fn parse_neg(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_neg({input:?})");
        alt((
            preceded(tag("!"), Self::parse_neg).map(Re::neg),
            Self::parse_kleene,
        ))
        .parse(input)
    }

    fn parse_kleene(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_kleene({input:?})");
        pair(Self::parse_or, many0_count(tag("*")))
            .map(|(mut res, num_stars)| {
                for _ in 0..num_stars {
                    res = Re::kleene(res);
                }
                res
            })
            .parse(input)
    }

    fn parse_or(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_or({input:?})");
        let (input, res) = Self::parse_and(input)?;
        fold_many0(
            preceded(tag("+"), Self::parse_and),
            move || res.clone(),
            Re::or,
        )
        .parse(input)
    }

    fn parse_and(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_and({input:?})");
        let (input, res) = Self::parse_parens(input)?;
        fold_many0(
            preceded(tag("&"), Self::parse_parens),
            move || res.clone(),
            Re::and,
        )
        .parse(input)
    }

    fn parse_parens(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_parens({input:?})");
        alt((
            delimited(tag("("), Self::parse_concat, tag(")")),
            Self::parse_literal,
        ))
        .parse(input)
    }

    fn parse_literal(input: &str) -> IResult<&str, Rc<Self>> {
        println!("parse_literal({input:?})");
        none_of("&+*()!").map(Re::char).parse(input)
    }
}

// Derivatives
impl Re {
    // TODO: test
    pub fn deriv(self: &Rc<Self>, c: char) -> Rc<Re> {
        match self.as_ref() {
            Re::Null => Rc::clone(self),
            Re::Empty => Re::null(),
            Re::Char(c2) if c == *c2 => Re::empty(),
            Re::Char(c2) => Re::null(),
            Re::Concat(lhs, rhs) => {
                if lhs.is_nullable() {
                    Re::or(lhs.deriv(c), rhs.deriv(c))
                } else {
                    lhs.deriv(c)
                }
            }
            Re::Kleene(re) => Re::concat(re.deriv(c), re.clone()),
            Re::Or(lhs, rhs) => Re::or(lhs.deriv(c), rhs.deriv(c)),
            Re::And(lhs, rhs) => Re::and(lhs.deriv(c), rhs.deriv(c)),
            Re::Neg(re) => Re::neg(re.deriv(c)),
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
}

// Canonicalization
impl Re {
    /// Partially canonicalize this regular expression. Per the paper, this means that any given
    /// regular expression has a finite number of representations under this canonicalization,
    /// which is enough to conservatively group equivalent regular expressions.
    pub fn canonicalize(self: &Rc<Self>) -> Rc<Self> {
        match self.as_ref() {
            Re::Or(lhs, rhs) => {
                let (mut lhs, mut rhs) = (lhs.canonicalize(), rhs.canonicalize());
                // Resolve to left-associative.
                if let Re::Or(rlhs, rrhs) = rhs.as_ref() {
                    lhs = Re::or(lhs.clone(), rlhs.clone()).canonicalize();
                    rhs = rrhs.clone();
                } else
                // If branches are the same, no need for `Or`.
                if rhs == lhs {
                    return lhs.clone();
                } else if lhs.is_or() {
                    return Re::or(lhs, rhs);
                } else
                // Otherwise resolve in order.
                if rhs < lhs {
                    (rhs, lhs) = (lhs, rhs);
                }
                Re::or(lhs, rhs)
            }
            Re::Neg(re) => {
                // Resolve !!r -> r.
                let re = re.canonicalize();
                if let Re::Neg(inner) = re.as_ref() {
                    inner.clone()
                } else {
                    Re::neg(re)
                }
            }
            Re::Concat(lhs, rhs) => Re::concat(lhs.canonicalize(), rhs.canonicalize()),
            Re::Kleene(re) => Re::kleene(re.canonicalize()),
            Re::And(lhs, rhs) => Re::and(lhs.canonicalize(), rhs.canonicalize()),
            _ => self.clone(),
        }
    }
}

// Matching
impl Re {
    /// Determine whether the given input matches the regular expression; that is, whether
    /// it is in the language defined by the expression.
    pub fn match_slow(&self, input: impl AsRef<str>) -> bool {
        self.match_inner(input.as_ref())
    }

    fn match_inner(&self, input: &str) -> bool {
        match self {
            Re::Null => false,
            Re::Empty => input.is_empty(),
            Re::Char(c) => {
                // Match a single-character string containing 'c'.
                let mut chars = input.chars();
                if chars.next() != Some(*c) {
                    return false;
                }
                if chars.next().is_some() {
                    return false;
                }
                true
            }
            Re::Concat(lhs, rhs) => {
                for (l, r) in Self::all_splits(input) {
                    if lhs.match_inner(l) && rhs.match_inner(r) {
                        return true;
                    }
                }
                false
            }
            Re::Kleene(re) => {
                // Trivially match zero repetitions
                if input.is_empty() {
                    return true;
                }

                // Match a single non-empty repetition followed by a match of self.
                for (l, r) in Self::all_splits(input).skip(1) {
                    if re.match_inner(l) && self.match_inner(r) {
                        return true;
                    }
                }
                false
            }
            Re::Or(lhs, rhs) => lhs.match_inner(input) || rhs.match_inner(input),
            Re::And(lhs, rhs) => lhs.match_inner(input) && rhs.match_inner(input),
            Re::Neg(re) => !re.match_inner(input),
        }
    }

    /// Generate all possible divisions of input into pairs of substrings, including an empty
    /// prefix and an empty suffix.
    fn all_splits(input: &str) -> impl Iterator<Item = (&str, &str)> {
        input
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(input.len()))
            .map(|i| input.split_at(i))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Parsing

    macro_rules! parse_test {
        ( $name:ident ( $input:expr ), $re:expr ) => {
            #[test]
            fn $name() {
                assert_eq!(Re::parse($input), $re);
            }
        };
    }

    parse_test!(parse_empty(""), Re::empty());
    parse_test!(parse_char("a"), Re::char('a'));
    parse_test!(parse_concat("ab"), Re::concat(Re::char('a'), Re::char('b')));
    parse_test!(parse_neg_char("!a"), Re::neg(Re::char('a')));
    parse_test!(
        parse_neg_char_kleen("!a*"),
        Re::neg(Re::kleene(Re::char('a')))
    );
    parse_test!(
        parse_triple_kleen("a***"),
        Re::kleene(Re::kleene(Re::kleene(Re::char('a'))))
    );
    parse_test!(
        parse_triple_neg("!!!a"),
        Re::neg(Re::neg(Re::neg(Re::char('a'))))
    );
    parse_test!(
        parse_neg_char_concat("!ab"),
        Re::concat(Re::neg(Re::char('a')), Re::char('b'))
    );
    parse_test!(
        parse_or("a+b+c"),
        Re::or(Re::or(Re::char('a'), Re::char('b')), Re::char('c'))
    );

    parse_test!(
        parse_and_or("a+b&c+d"),
        Re::or(
            Re::or(Re::char('a'), Re::and(Re::char('b'), Re::char('c'))),
            Re::char('d')
        )
    );
    parse_test!(
        parse_parens("(a+b)&(c+d)"),
        Re::and(
            Re::or(Re::char('a'), Re::char('b')),
            Re::or(Re::char('c'), Re::char('d')),
        )
    );
    parse_test!(
        parse_parens_kleene("(abc)*"),
        Re::kleene(Re::concat(
            Re::concat(Re::char('a'), Re::char('b')),
            Re::char('c')
        ))
    );

    // Matching

    macro_rules! match_test {
        ( $name:ident ( $re:expr, $input:expr ), $matches:ident ) => {
            #[test]
            fn $name() {
                assert_eq!(dbg!(Re::parse($re)).match_slow($input), $matches);
            }
        };
    }

    match_test!(match_empty("", ""), true);
    match_test!(match_nonempty("", "abc"), false);
    match_test!(match_char("a", "a"), true);
    match_test!(match_char_different("x", "a"), false);
    match_test!(match_char_empty("a", ""), false);
    match_test!(match_concat("ab", "ab"), true);
    match_test!(match_concat_extra("ab", "abc"), false);
    match_test!(match_concat_short("abc", "ab"), false);
    match_test!(match_concat_different("xyz", "abc"), false);
    match_test!(match_concat_suffix("abc", "xabc"), false);
    match_test!(match_kleene_none("a*", ""), true);
    match_test!(match_kleene_one("a*", "a"), true);
    match_test!(match_kleene_two("a*", "aa"), true);
    match_test!(match_double_kleene_zero("a*b*", "bbbb"), true);
    match_test!(match_double_kleene_one("a*b*", "abbb"), true);
    match_test!(match_double_kleene_two("a*b*", "aabb"), true);
    match_test!(match_double_kleene_three("a*b*", "aaab"), true);
    match_test!(match_double_kleene_four("a*b*", "aaaa"), true);
    match_test!(match_kleene_alternation("(ab)*", "ababab"), true);
    match_test!(match_kleene_alternation_short("(ab)*", "ababa"), false);
    match_test!(match_optional("x((abc)+())y", "xabcy"), true);
    match_test!(match_optional_missing("x((abc)+())y", "xy"), true);
    match_test!(match_neg("x!(abc)y", "xy"), true);
    match_test!(match_neg_less("x!(abc)y", "xaby"), true);
    match_test!(match_neg_more("x!(abc)y", "xabcdy"), true);
    match_test!(match_neg_false("x!(abc)y", "xabcy"), false);

    // Canonicalization

    macro_rules! canon_test {
        ( $name:ident ( $re:expr, $canonical:expr ) ) => {
            #[test]
            fn $name() {
                assert_eq!($re.canonicalize(), $canonical);
                assert_eq!($canonical.canonicalize(), $canonical);
            }
        };
    }

    canon_test!(collapse_or(
        Re::or(Re::char('a'), Re::char('a')),
        Re::char('a')
    ));

    canon_test!(left_assoc_or(
        Re::neg(Re::or(Re::char('a'), Re::or(Re::char('b'), Re::char('c')))),
        Re::neg(Re::or(Re::or(Re::char('a'), Re::char('b')), Re::char('c')))
    ));

    canon_test!(left_assoc_or_multi(
        Re::neg(Re::or(
            Re::char('a'),
            Re::or(Re::char('b'), Re::or(Re::char('c'), Re::char('d')))
        )),
        Re::neg(Re::or(
            Re::or(Re::or(Re::char('a'), Re::char('b')), Re::char('c')),
            Re::char('d')
        ))
    ));
}
