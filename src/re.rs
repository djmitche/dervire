use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::none_of,
    multi::{fold_many0, many0_count},
    sequence::{delimited, pair, preceded},
};
use std::{cell::OnceCell, rc::Rc};

#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Hash)]
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
    #[allow(clippy::should_implement_trait)]
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
        alt((
            preceded(tag("!"), Self::parse_neg).map(Re::neg),
            Self::parse_kleene,
        ))
        .parse(input)
    }

    fn parse_kleene(input: &str) -> IResult<&str, Rc<Self>> {
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
        let (input, res) = Self::parse_and(input)?;
        fold_many0(
            preceded(tag("+"), Self::parse_and),
            move || res.clone(),
            Re::or,
        )
        .parse(input)
    }

    fn parse_and(input: &str) -> IResult<&str, Rc<Self>> {
        let (input, res) = Self::parse_parens(input)?;
        fold_many0(
            preceded(tag("&"), Self::parse_parens),
            move || res.clone(),
            Re::and,
        )
        .parse(input)
    }

    fn parse_parens(input: &str) -> IResult<&str, Rc<Self>> {
        alt((
            delimited(tag("("), Self::parse_concat, tag(")")),
            Self::parse_literal,
        ))
        .parse(input)
    }

    fn parse_literal(input: &str) -> IResult<&str, Rc<Self>> {
        none_of("&+*()!").map(Re::char).parse(input)
    }
}

// Derivatives
impl Re {
    pub fn deriv(self: &Rc<Self>, c: char) -> Rc<Re> {
        match self.as_ref() {
            Re::Null => Rc::clone(self),
            Re::Empty => Re::null(),
            Re::Char(c2) if c == *c2 => Re::empty(),
            Re::Char(c2) => Re::null(),
            Re::Concat(lhs, rhs) => Re::or(
                Re::concat(lhs.deriv(c), rhs.clone()),
                Re::concat(lhs.v(), rhs.deriv(c)),
            ),
            Re::Kleene(re) => Re::concat(re.deriv(c), self.clone()),
            Re::Or(lhs, rhs) => Re::or(lhs.deriv(c), rhs.deriv(c)),
            Re::And(lhs, rhs) => Re::and(lhs.deriv(c), rhs.deriv(c)),
            Re::Neg(re) => Re::neg(re.deriv(c)),
        }
    }

    /// Returns true if this regular expression matches the empty string.
    pub fn is_nullable(self: &Rc<Self>) -> bool {
        self.v().is_empty()
    }

    // Defined in section 3.1.
    fn v(&self) -> Rc<Re> {
        match self {
            Re::Empty => Re::empty(),
            Re::Char(_) => Re::null(),
            Re::Null => Re::null(),
            Re::Concat(lhs, rhs) => Re::and(lhs.v(), rhs.v()),
            Re::Kleene(_) => Re::empty(),
            Re::Or(lhs, rhs) => Re::or(lhs.v(), rhs.v()),
            Re::And(lhs, rhs) => Re::and(lhs.v(), rhs.v()),
            Re::Neg(re) => match re.v().as_ref() {
                Re::Null => Re::empty(),
                Re::Empty => Re::null(),
                _ => unreachable!(),
            },
        }
        .canonicalize()
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
                match (lhs.as_ref(), rhs.as_ref()) {
                    (Re::Null, _) => rhs,
                    (_, Re::Null) => lhs,
                    // Remove duplicates
                    (_, _) if lhs == rhs => lhs,
                    // Resolve to left-associative.
                    (_, Re::Or(rlhs, rrhs)) => {
                        lhs = Re::or(lhs.clone(), rlhs.clone()).canonicalize();
                        rhs = rrhs.clone();
                        Re::or(lhs, rhs)
                    }
                    // If already left-associative, leave it
                    (Re::Or(_, _), _) => Re::or(lhs, rhs),
                    // Return in sorted order
                    (_, _) if lhs > rhs => Re::or(rhs, lhs),
                    (_, _) => Re::or(lhs, rhs),
                }
            }
            Re::And(lhs, rhs) => {
                let (mut lhs, mut rhs) = (lhs.canonicalize(), rhs.canonicalize());
                match (lhs.as_ref(), rhs.as_ref()) {
                    (Re::Null, _) => Re::null(),
                    (_, Re::Null) => Re::null(),
                    // Remove duplicates
                    (_, _) if lhs == rhs => lhs,
                    // Resolve to left-associative.
                    (_, Re::And(rlhs, rrhs)) => {
                        lhs = Re::and(lhs.clone(), rlhs.clone()).canonicalize();
                        rhs = rrhs.clone();
                        Re::and(lhs, rhs)
                    }
                    // If already left-associative, leave it
                    (Re::And(_, _), _) => Re::and(lhs, rhs),
                    // Return in sorted order
                    (_, _) if lhs > rhs => Re::and(rhs, lhs),
                    (_, _) => Re::and(lhs, rhs),
                }
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
            Re::Concat(lhs, rhs) => {
                let (mut lhs, mut rhs) = (lhs.canonicalize(), rhs.canonicalize());
                match (lhs.as_ref(), rhs.as_ref()) {
                    (Re::Null, _) => Re::null(),
                    (_, Re::Null) => Re::null(),
                    (Re::Empty, _) => rhs,
                    (_, Re::Empty) => lhs,
                    // Resolve to left-associative.
                    (_, Re::Concat(rlhs, rrhs)) => {
                        lhs = Re::concat(lhs.clone(), rlhs.clone()).canonicalize();
                        rhs = rrhs.clone();
                        Re::concat(lhs, rhs)
                    }
                    (_, _) => Re::concat(lhs, rhs),
                }
            }
            Re::Kleene(re) => Re::kleene(re.canonicalize()),
            _ => self.clone(),
        }
    }
}

// Matching
impl Re {
    /// Determine whether the given input matches the regular expression; that is, whether
    /// it is in the language defined by the expression.
    pub fn matches_slow(&self, input: impl AsRef<str>) -> bool {
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

    // Derivatives

    macro_rules! deriv_test {
        ( $name:ident ( $input:expr, $char:expr ), $output:expr ) => {
            #[test]
            fn $name() {
                assert_eq!($input.deriv($char).canonicalize(), $output);
            }
        };
    }

    deriv_test!(deriv_null(Re::null(), 'a'), Re::null());
    deriv_test!(deriv_empty(Re::empty(), 'a'), Re::null());
    deriv_test!(deriv_matching_char(Re::char('a'), 'a'), Re::empty());
    deriv_test!(deriv_nonmatching_char(Re::char('a'), 'x'), Re::null());
    deriv_test!(
        deriv_concat(Re::concat(Re::char('a'), Re::char('b')), 'a'),
        Re::char('b')
    );
    deriv_test!(
        deriv_concat_nonmatching(Re::concat(Re::char('a'), Re::char('b')), 'x'),
        Re::null()
    );
    deriv_test!(
        deriv_concat_nullable(Re::concat(Re::kleene(Re::char('a')), Re::char('b')), 'a'),
        Re::concat(Re::kleene(Re::char('a')), Re::char('b'))
    );
    deriv_test!(
        deriv_matching_kleene(Re::kleene(Re::char('a')), 'a'),
        Re::kleene(Re::char('a'))
    );
    deriv_test!(
        deriv_nonmatching_kleene(Re::kleene(Re::char('a')), 'x'),
        Re::null()
    );
    deriv_test!(
        deriv_or_both_match(
            Re::or(
                Re::concat(Re::char('a'), Re::char('b')),
                Re::concat(Re::char('a'), Re::char('c'))
            ),
            'a'
        ),
        Re::or(Re::char('b'), Re::char('c'))
    );
    deriv_test!(
        deriv_or_one_matches(
            Re::or(
                Re::concat(Re::char('a'), Re::char('b')),
                Re::concat(Re::char('c'), Re::char('d'))
            ),
            'a'
        ),
        Re::char('b')
    );
    deriv_test!(
        deriv_and_both_match(
            Re::and(
                Re::concat(Re::char('a'), Re::char('b')),
                Re::concat(Re::char('a'), Re::char('c'))
            ),
            'a'
        ),
        Re::and(Re::char('b'), Re::char('c'))
    );
    deriv_test!(
        deriv_and_one_matches(
            Re::and(
                Re::concat(Re::char('a'), Re::char('b')),
                Re::concat(Re::char('c'), Re::char('d'))
            ),
            'a'
        ),
        Re::null()
    );

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

    canon_test!(canon_or_collapse_equal(
        Re::or(Re::char('a'), Re::char('a')),
        Re::char('a')
    ));

    canon_test!(canon_or_sorted(
        Re::or(Re::char('b'), Re::char('a')),
        Re::or(Re::char('a'), Re::char('b'))
    ));

    canon_test!(canon_or_collapse_null_lhs(
        Re::or(Re::null(), Re::char('a')),
        Re::char('a')
    ));

    canon_test!(canon_or_collapse_null_rhs(
        Re::or(Re::char('a'), Re::null()),
        Re::char('a')
    ));

    canon_test!(canon_or_left_assoc(
        Re::neg(Re::or(Re::char('a'), Re::or(Re::char('b'), Re::char('c')))),
        Re::neg(Re::or(Re::or(Re::char('a'), Re::char('b')), Re::char('c')))
    ));

    canon_test!(canon_or_left_assoc_multi(
        Re::neg(Re::or(
            Re::char('a'),
            Re::or(Re::char('b'), Re::or(Re::char('c'), Re::char('d')))
        )),
        Re::neg(Re::or(
            Re::or(Re::or(Re::char('a'), Re::char('b')), Re::char('c')),
            Re::char('d')
        ))
    ));

    canon_test!(canon_and_collapse_equal(
        Re::and(Re::char('a'), Re::char('a')),
        Re::char('a')
    ));

    canon_test!(canon_and_sorted(
        Re::and(Re::char('b'), Re::char('a')),
        Re::and(Re::char('a'), Re::char('b'))
    ));

    canon_test!(canon_and_collapse_null_lhs(
        Re::and(Re::null(), Re::char('a')),
        Re::null()
    ));

    canon_test!(canon_and_collapse_null_rhs(
        Re::and(Re::char('a'), Re::null()),
        Re::null()
    ));

    canon_test!(canon_and_left_assoc(
        Re::neg(Re::and(
            Re::char('a'),
            Re::and(Re::char('b'), Re::char('c'))
        )),
        Re::neg(Re::and(
            Re::and(Re::char('a'), Re::char('b')),
            Re::char('c')
        ))
    ));

    canon_test!(canon_and_left_assoc_multi(
        Re::neg(Re::and(
            Re::char('a'),
            Re::and(Re::char('b'), Re::and(Re::char('c'), Re::char('d')))
        )),
        Re::neg(Re::and(
            Re::and(Re::and(Re::char('a'), Re::char('b')), Re::char('c')),
            Re::char('d')
        ))
    ));

    canon_test!(canon_concat_no_collapse_equal(
        Re::concat(Re::char('a'), Re::char('a')),
        Re::concat(Re::char('a'), Re::char('a'))
    ));

    canon_test!(canon_concat_not_sorted(
        Re::concat(Re::char('b'), Re::char('a')),
        Re::concat(Re::char('b'), Re::char('a'))
    ));

    canon_test!(canon_concat_collapse_null_lhs(
        Re::concat(Re::null(), Re::char('a')),
        Re::null()
    ));

    canon_test!(canon_concat_collapse_empty_lhs(
        Re::concat(Re::empty(), Re::char('a')),
        Re::char('a')
    ));

    canon_test!(canon_concat_collapse_null_rhs(
        Re::concat(Re::char('a'), Re::null()),
        Re::null()
    ));

    canon_test!(canon_concat_collapse_empty_rhs(
        Re::concat(Re::char('a'), Re::empty()),
        Re::char('a')
    ));

    canon_test!(canon_concat_left_assoc(
        Re::neg(Re::concat(
            Re::char('a'),
            Re::concat(Re::char('b'), Re::char('c'))
        )),
        Re::neg(Re::concat(
            Re::concat(Re::char('a'), Re::char('b')),
            Re::char('c')
        ))
    ));

    canon_test!(canon_concat_left_assoc_multi(
        Re::neg(Re::concat(
            Re::char('a'),
            Re::concat(Re::char('b'), Re::concat(Re::char('c'), Re::char('d')))
        )),
        Re::neg(Re::concat(
            Re::concat(Re::concat(Re::char('a'), Re::char('b')), Re::char('c')),
            Re::char('d')
        ))
    ));

    canon_test!(canon_neg_double(
        Re::neg(Re::neg(Re::char('a'))),
        Re::char('a')
    ));

    // Matching

    fn match_test(re: &str, input: &str) -> bool {
        dbg!(Re::parse(re)).matches_slow(input)
    }
    crate::test_macros::match_tests!(match_test);
}
