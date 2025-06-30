const X: u32 = 10;
macro_rules! match_test {
    ( $t:ident, $name:ident ( $re:expr, $input:expr ), $matches:ident ) => {
        #[test]
        fn $name() {
            assert_eq!($t($re, $input), $matches);
        }
    };
}
pub(crate) use match_test;

/// Define a collection of tests for matching regular expressions.
macro_rules! match_tests {
    ($t:ident) => {
        mod matching {
            #[allow(unused_imports)]
            pub use super::*;
            use $crate::test_macros::match_test as t;
            t!($t, empty("", ""), true);
            t!($t, nonempty("", "abc"), false);
            t!($t, char("a", "a"), true);
            t!($t, char_different("x", "a"), false);
            t!($t, char_empty("a", ""), false);
            t!($t, concat("ab", "ab"), true);
            t!($t, concat_extra("ab", "abc"), false);
            t!($t, concat_short("abc", "ab"), false);
            t!($t, concat_different("xyz", "abc"), false);
            t!($t, concat_suffix("abc", "xabc"), false);
            t!($t, kleene_none("a*", ""), true);
            t!($t, kleene_one("a*", "a"), true);
            t!($t, kleene_two("a*", "aa"), true);
            t!($t, double_kleene_zero("a*b*", "bbbb"), true);
            t!($t, double_kleene_one("a*b*", "abbb"), true);
            t!($t, double_kleene_two("a*b*", "aabb"), true);
            t!($t, double_kleene_three("a*b*", "aaab"), true);
            t!($t, double_kleene_four("a*b*", "aaaa"), true);
            t!($t, kleene_alternation("(ab)*", "ababab"), true);
            t!($t, kleene_alternation_short("(ab)*", "ababa"), false);
            t!($t, kleene_alternation_multi_matches("(a*b*)*", "abaaababbb"), true);
            t!($t, optional("x((abc)+())y", "xabcy"), true);
            t!($t, optional_missing("x((abc)+())y", "xy"), true);
            t!($t, neg("x!(abc)y", "xy"), true);
            t!($t, neg_less("x!(abc)y", "xaby"), true);
            t!($t, neg_more("x!(abc)y", "xabcdy"), true);
            t!($t, neg_false("x!(abc)y", "xabcy"), false);
        }
    };
}
pub(crate) use match_tests;
