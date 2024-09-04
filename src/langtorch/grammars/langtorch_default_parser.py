from pyparsing import *
import logging
import re

LBRACE, RBRACE, COLON, BACKTICK = map(Suppress, '{}:`')
value = CharsNotIn('{}:`')
value_w_colon = CharsNotIn('{}`')
value_backticked = CharsNotIn('`')
key = CharsNotIn('{}', min=1)

# Modified empty string handling
empty_string = White(min=1).setParseAction(lambda t: [('', t[0])])

# Existing patterns (unchanged)
unnamed_string1 = (LBRACE + value("value") + (Optional(COLON) ^ StringEnd()) + RBRACE)
unnamed_string2 = (LBRACE + BACKTICK + value("value") + BACKTICK + COLON + RBRACE)
unnamed_string3 = (LBRACE + BACKTICK + value_backticked("value") + BACKTICK + RBRACE)
unnamed_string4 = (LBRACE + BACKTICK + value_backticked("value") + BACKTICK + COLON + RBRACE)
empty_unnamed_string1 = (LBRACE + RBRACE).setParseAction(lambda t: [('', '')])
empty_unnamed_string2 = (BACKTICK + BACKTICK).setParseAction(lambda t: [('', '')])
unnamed_string5 = value_w_colon("value")
unnamed_string6 = BACKTICK + value_backticked("value") + BACKTICK

named_string1 = Group(value("value") + LBRACE + COLON + key("key") + RBRACE)
named_string2 = Group(BACKTICK + value("value") + BACKTICK + LBRACE + COLON + key("key") + RBRACE)
named_string3 = Group(LBRACE + value("value") + COLON + key("key") + RBRACE)
named_string4 = Group(LBRACE + BACKTICK + value("value") + BACKTICK + COLON + key("key") + RBRACE)
named_string5 = Group(value("value") + LBRACE + BACKTICK + BACKTICK + COLON + RBRACE)
empty_named_string = (LBRACE + COLON + RBRACE).setParseAction(lambda t: [('', '')])
backticked_empty_key = Group(LBRACE + BACKTICK + BACKTICK + COLON + key("key") + RBRACE).setParseAction(lambda t: [(t[0]['key'], '')])

# Grouping the unnamed string patterns
unnamed_string = (unnamed_string1
                  | unnamed_string2
                  | unnamed_string3
                  | unnamed_string4
                  | empty_unnamed_string1
                  | unnamed_string5
                  | unnamed_string6
                  | empty_unnamed_string2
                  | empty_string)  # This will now catch spaces both inside and outside brackets

# Grouping the named string patterns (unchanged)
named_string = (empty_named_string
                | named_string1
                | named_string2
                | named_string3
                | named_string4
                | backticked_empty_key
                | named_string5)

# Constructing the final parser pattern
LangTorchGrammarParser = ZeroOrMore(named_string | unnamed_string) + StringEnd()

def fix_double_brackets(s):
    # Regex Explanation:
    # Add that 0 or 1 $ symbol can appear between both  left and right {${
    # (?<!\{) - Negative lookbehind to ensure no '{' immediately before our pattern
    # {{ - Matches '{{'
    # ([^{}]+) - Captures one or more characters that are not '{' or '}'
    # }} - Matches '}}'
    # (?![^{]*\}) - Negative lookahead to ensure our pattern is not within an outer '{...}'
    pattern = r'(?<!\{){{([^{}]+)}}(?![^{]*\})'

    # Replacement pattern
    # We use a lambda to format the replacement string with backticks around the captured group
    replacement = lambda m: '{`{' + m.group(1) + '}`}'

    # Substitute using the pattern and replacement
    result = re.sub(pattern, replacement, s)

    return result
def fix_substitution(s):
    pattern = r'(?<!\$)\$\{([^{}]+)\}(?![^{]*\})'
    replacement = lambda m: '{$' + m.group(1) + '}'
    result = re.sub(pattern, replacement, s)
    return result

def fix_backslashed(s):
    def is_within_backticks(s, start, end):
        """Check if the substring from start to end is within backticks in the original string."""
        backtick_pairs = list(re.finditer(r'`[^`]*`', s))
        for pair in backtick_pairs:
            if pair.start() <= start and end <= pair.end():
                return True
        return False

    def replace_func(match):
        start, end = match.span()

        if is_within_backticks(s, start, end):
            return match.group(0)  # Return the original match if it's within backticks
        else:
            f"`{{{match.group(1)}}}`"

    pattern = r'\\{(.*?)\\}'

    return re.sub(pattern, replace_func, s)

def fix_spaces_between_brackets(s):
    def is_within_backticks(s, start, end):
        """Check if the substring from start to end is within backticks in the original string."""
        backtick_pairs = list(re.finditer(r'`[^`]*`', s))
        for pair in backtick_pairs:
            if pair.start() <= start and end <= pair.end():
                return True
        return False

    pattern = r'\}(\s+)\{'

    def replace_func(match):
        whitespace = match.group(1)
        start, end = match.span()

        if is_within_backticks(s, start, end):
            return match.group(0)  # Return the original match if it's within backticks
        else:
            return f'}}`{whitespace}`{{'  # Add backticks around the whitespace

    return re.sub(pattern, replace_func, s)

# Test function
def test_parser(test_strings):
    for test_string in test_strings:
        print(f"Testing: {test_string}")
        try:
            result = LangTorchGrammarParser.parseString((fix_spaces_between_brackets(test_string)))
            print(f"Parsed: {result.asList()}")
        except ParseException as e:
            print(f"Parse error: {e}")
        print()

# # Test cases
# test_strings = [
#     "Simple text",
#     "{}  {}",
#     "Text with {variable}",
#     "Text with {} this should parse into three parts with the second being an empty string",
#     "Text with {variable:key}",
#     "Text with `backticks` that are just ignored",
#     "Text with {`normal value in backticks`}",
#     "Text with `{unparsed value in backticks}`",
#     "Text with {`backticked value`:key}",
#     "Text with escaped \\{ and \\}",
#     "Text with empty element {} and value{:with key for the text before}",
#     "Text with {``:empty value with key}",
#     "Text with multiple {var1} and {var2:key2}",
# ]
#
# # Run the tests
# test_parser(test_strings)
#
#
# BNF = """TextParser = { (NamedString | UnnamedString) } ;
#
# (* Unnamed strings *)
#
# UnnamedString = UnnamedString1
#               | UnnamedString2
#               | UnnamedString3
#               | UnnamedString4
#               | EmptyUnnamedString1
#               | UnnamedString5
#               | UnnamedString6
#               | EmptyUnnamedString2
#               ;
#
# UnnamedString1 = LBRACE, VALUE, [COLON], RBRACE ;
# UnnamedString2 = LBRACE, BACKTICK, VALUE, BACKTICK, COLON, RBRACE ;
# UnnamedString3 = LBRACE, BACKTICK, VALUE, BACKTICK, RBRACE ;
# UnnamedString4 = LBRACE, BACKTICK, VALUE, BACKTICK, COLON, RBRACE ;
# UnnamedString5 = VALUE ;
# UnnamedString6 = BACKTICK, VALUE, BACKTICK ;
# EmptyUnnamedString1 = LBRACE, RBRACE ;
# EmptyUnnamedString2 = BACKTICK, BACKTICK ;
#
# (* Named strings *)
#
# NamedString = NamedString1
#             | NamedString2
#             | NamedString3
#             | NamedString4
#             | NamedString5
#             | EmptyNamedString
#             | BacktickedEmptyKey
#             ;
#
# NamedString1 = VALUE, LBRACE, COLON, VALUE, RBRACE ;
# NamedString2 = BACKTICK, VALUE, BACKTICK, LBRACE, COLON, VALUE, RBRACE ;
# NamedString3 = LBRACE, VALUE, COLON, VALUE, RBRACE ;
# NamedString4 = LBRACE, BACKTICK, VALUE, BACKTICK, COLON, VALUE, RBRACE ;
# NamedString5 = VALUE, LBRACE, BACKTICK, BACKTICK, COLON, RBRACE ;
# EmptyNamedString = LBRACE, COLON, RBRACE ;
# BacktickedEmptyKey = LBRACE, BACKTICK, BACKTICK, COLON, VALUE, RBRACE ;
# """
