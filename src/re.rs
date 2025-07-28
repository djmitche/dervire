use regex_syntax;
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

// Parsing using regex-syntax for standard regex patterns
impl Re {
    /// Parse standard regex syntax using regex-syntax crate
    pub fn parse(pattern: impl AsRef<str>) -> Rc<Self> {
        let hir = regex_syntax::parse(pattern.as_ref()).unwrap_or_else(|e| {
            panic!("Invalid regular expression {}: {}", pattern.as_ref(), e)
        });
        Self::from_hir(&hir)
    }

    /// Convert regex-syntax HIR to our internal Re representation
    fn from_hir(hir: &regex_syntax::hir::Hir) -> Rc<Self> {
        use regex_syntax::hir::HirKind;
        
        match hir.kind() {
            HirKind::Empty => Re::empty(),
            HirKind::Literal(literal) => {
                // Convert bytes to string and then to chars
                let bytes = &literal.0;
                if let Ok(s) = std::str::from_utf8(bytes) {
                    let chars: Vec<char> = s.chars().collect();
                    if chars.is_empty() {
                        Re::empty()
                    } else if chars.len() == 1 {
                        Re::char(chars[0])
                    } else {
                        chars.into_iter()
                            .map(Re::char)
                            .reduce(Re::concat)
                            .unwrap_or_else(Re::empty)
                    }
                } else {
                    // If it's not valid UTF-8, use a placeholder
                    Re::char('_')
                }
            }
            HirKind::Class(class) => {
                // Convert character class to alternation
                use regex_syntax::hir::Class;
                match class {
                    Class::Unicode(unicode_class) => {
                        let chars: Vec<Rc<Re>> = unicode_class.iter()
                            .flat_map(|range| {
                                (range.start()..=range.end()).map(Re::char)
                            })
                            .collect();
                        
                        if chars.is_empty() {
                            Re::null()
                        } else if chars.len() == 1 {
                            chars.into_iter().next().unwrap()
                        } else {
                            chars.into_iter().reduce(Re::or).unwrap_or_else(Re::null)
                        }
                    }
                    Class::Bytes(_) => {
                        // For byte classes, use a placeholder
                        Re::char('_')
                    }
                }
            }
            HirKind::Look(_) => {
                // Look-around assertions - simplified to empty for this demo
                Re::empty()
            }
            HirKind::Repetition(rep) => {
                let inner = Self::from_hir(&rep.sub);
                // Handle repetition based on min/max bounds
                match (rep.min, rep.max) {
                    (0, None) => Re::kleene(inner), // {0,} = *
                    (1, None) => Re::concat(inner.clone(), Re::kleene(inner)), // {1,} = +
                    (0, Some(1)) => Re::or(Re::empty(), inner), // {0,1} = ?
                    (0, Some(0)) => Re::empty(), // {0,0} = ε
                    _ => inner, // Simplification for other ranges
                }
            }
            HirKind::Capture(capture) => {
                // Ignore capture groups and just process the inner expression
                Self::from_hir(&capture.sub)
            }
            HirKind::Concat(parts) => {
                if parts.is_empty() {
                    Re::empty()
                } else {
                    parts.iter()
                        .map(Self::from_hir)
                        .reduce(Re::concat)
                        .unwrap_or_else(Re::empty)
                }
            }
            HirKind::Alternation(parts) => {
                if parts.is_empty() {
                    Re::null()
                } else {
                    parts.iter()
                        .map(Self::from_hir)
                        .reduce(Re::or)
                        .unwrap_or_else(Re::null)
                }
            }
        }
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

    // Test regex-syntax based parser
    #[test]
    fn test_regex_parsing() {
        // Test simple literal
        let re = Re::parse("a");
        assert!(re.matches_slow("a"));
        assert!(!re.matches_slow("b"));

        // Test concatenation  
        let re = Re::parse("ab");
        assert!(re.matches_slow("ab"));
        assert!(!re.matches_slow("a"));

        // Test alternation
        let re = Re::parse("a|b");
        assert!(re.matches_slow("a"));
        assert!(re.matches_slow("b"));
        assert!(!re.matches_slow("c"));

        // Test Kleene star
        let re = Re::parse("a*");
        assert!(re.matches_slow(""));
        assert!(re.matches_slow("a"));
        assert!(re.matches_slow("aa"));

        // Test plus (one or more)
        let re = Re::parse("a+");
        assert!(!re.matches_slow(""));
        assert!(re.matches_slow("a"));
        assert!(re.matches_slow("aa"));

        // Test question mark (zero or one)
        let re = Re::parse("a?");
        assert!(re.matches_slow(""));
        assert!(re.matches_slow("a"));
        assert!(!re.matches_slow("aa"));
    }

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

    // Matching - Updated to use standard regex syntax
    fn match_test(re: &str, input: &str) -> bool {
        dbg!(Re::parse(re)).matches_slow(input)
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
}
