from pyparsing import *

escaped_char = Suppress("\\") + oneOf("{}:`")
LBRACE, RBRACE, COLON, BACKTICK = map(Suppress, '{}:`')
value = CharsNotIn('{}:`')
value_w_colon = CharsNotIn('{}`')
value_backticked = CharsNotIn('`')
key = CharsNotIn('{}', min=1)  # Ensure key has at least one character

# Redefining unnamed string patterns with improved backtick handling
unnamed_string1 = (LBRACE + value("value") + (Optional(COLON) ^ StringEnd()) + RBRACE)
unnamed_string2 = (LBRACE + BACKTICK + value("value") + BACKTICK + COLON + RBRACE)
unnamed_string3 = (LBRACE + BACKTICK + value("value") + BACKTICK + RBRACE)
unnamed_string4 = (LBRACE + BACKTICK + value("value") + BACKTICK + COLON + RBRACE)
empty_unnamed_string1 = (LBRACE + RBRACE).setParseAction(lambda t: '')
empty_unnamed_string2 = (BACKTICK + BACKTICK).setParseAction(lambda t: '')
unnamed_string5 = value_w_colon("value")
unnamed_string6 = BACKTICK + value_backticked("value") + BACKTICK

# Redefining the named string patterns with improved backtick handling
# Patterns for named strings

named_string1 = Group(value("value") + LBRACE + COLON + key("key") + RBRACE)
named_string2 = Group(BACKTICK + value("value") + BACKTICK + LBRACE + COLON + key("key") + RBRACE)
named_string3 = Group(LBRACE + value("value") + COLON + key("key") + RBRACE)
named_string4 = Group(LBRACE + BACKTICK + value("value") + BACKTICK + COLON + key("key") + RBRACE)
named_string5 = Group(value("value") + LBRACE + BACKTICK + BACKTICK + COLON + RBRACE)
empty_named_string = (LBRACE + COLON + RBRACE).setParseAction(lambda t: ('', ''))

# Patterns for empty strings and backticked content
backticked_empty_key = Group(LBRACE + BACKTICK + BACKTICK + COLON + key("key") + RBRACE)

# Grouping the unnamed string patterns with the new patterns
unnamed_string = (unnamed_string1
                  | unnamed_string2
                  | unnamed_string3
                  | unnamed_string4
                  | empty_unnamed_string1
                  | unnamed_string5
                  | unnamed_string6
                  | empty_unnamed_string2)

# Grouping the named string patterns with the new pattern for backticked empty key
named_string = (empty_named_string
                | named_string1
                | named_string2
                | named_string3
                | named_string4
                | backticked_empty_key
                | named_string5)

# Constructing the final parser pattern with preference for unnamed strings
LangTorchGrammarParser = ZeroOrMore(named_string | unnamed_string) + StringEnd()


def LangTorch_f_string(txt):
    parsed_result = LangTorchGrammarParser.parseString(txt)
    items = [(res.key if "key" in res else "", res.value if "value" in res else "") if isinstance(res,
                                                                                                  ParseResults) else res
             for res in parsed_result]
    return items


BNF = """TextParser = { (NamedString | UnnamedString) } ;

(* Unnamed strings *)

UnnamedString = UnnamedString1
              | UnnamedString2
              | UnnamedString3
              | UnnamedString4
              | EmptyUnnamedString1
              | UnnamedString5
              | UnnamedString6
              | EmptyUnnamedString2
              ;

UnnamedString1 = LBRACE, VALUE, [COLON], RBRACE ;
UnnamedString2 = LBRACE, BACKTICK, VALUE, BACKTICK, COLON, RBRACE ;
UnnamedString3 = LBRACE, BACKTICK, VALUE, BACKTICK, RBRACE ;
UnnamedString4 = LBRACE, BACKTICK, VALUE, BACKTICK, COLON, RBRACE ;
UnnamedString5 = VALUE ;
UnnamedString6 = BACKTICK, VALUE, BACKTICK ;
EmptyUnnamedString1 = LBRACE, RBRACE ;
EmptyUnnamedString2 = BACKTICK, BACKTICK ;

(* Named strings *)

NamedString = NamedString1
            | NamedString2
            | NamedString3
            | NamedString4
            | NamedString5
            | EmptyNamedString
            | BacktickedEmptyKey
            ;

NamedString1 = VALUE, LBRACE, COLON, VALUE, RBRACE ;
NamedString2 = BACKTICK, VALUE, BACKTICK, LBRACE, COLON, VALUE, RBRACE ;
NamedString3 = LBRACE, VALUE, COLON, VALUE, RBRACE ;
NamedString4 = LBRACE, BACKTICK, VALUE, BACKTICK, COLON, VALUE, RBRACE ;
NamedString5 = VALUE, LBRACE, BACKTICK, BACKTICK, COLON, RBRACE ;
EmptyNamedString = LBRACE, COLON, RBRACE ;
BacktickedEmptyKey = LBRACE, BACKTICK, BACKTICK, COLON, VALUE, RBRACE ;
"""
