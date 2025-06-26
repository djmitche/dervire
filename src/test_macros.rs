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
        use $crate::test_macros::match_test;
        match_test!($t, empty("", ""), true);
        match_test!($t, nonempty("", "abc"), false);
        match_test!($t, char("a", "a"), true);
        match_test!($t, char_different("x", "a"), false);
        match_test!($t, char_empty("a", ""), false);
        match_test!($t, concat("ab", "ab"), true);
        match_test!($t, concat_extra("ab", "abc"), false);
        match_test!($t, concat_short("abc", "ab"), false);
        match_test!($t, concat_different("xyz", "abc"), false);
        match_test!($t, concat_suffix("abc", "xabc"), false);
        match_test!($t, kleene_none("a*", ""), true);
        match_test!($t, kleene_one("a*", "a"), true);
        match_test!($t, kleene_two("a*", "aa"), true);
        match_test!($t, double_kleene_zero("a*b*", "bbbb"), true);
        match_test!($t, double_kleene_one("a*b*", "abbb"), true);
        match_test!($t, double_kleene_two("a*b*", "aabb"), true);
        match_test!($t, double_kleene_three("a*b*", "aaab"), true);
        match_test!($t, double_kleene_four("a*b*", "aaaa"), true);
        match_test!($t, kleene_alternation("(ab)*", "ababab"), true);
        match_test!($t, kleene_alternation_short("(ab)*", "ababa"), false);
        match_test!($t, optional("x((abc)+())y", "xabcy"), true);
        match_test!($t, optional_missing("x((abc)+())y", "xy"), true);
        match_test!($t, neg("x!(abc)y", "xy"), true);
        match_test!($t, neg_less("x!(abc)y", "xaby"), true);
        match_test!($t, neg_more("x!(abc)y", "xabcdy"), true);
        match_test!($t, neg_false("x!(abc)y", "xabcy"), false);
        }
    };
}
pub(crate) use match_tests;
