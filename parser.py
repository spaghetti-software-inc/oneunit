#!/usr/bin/env python3

"""
A combined lexer and recursive-descent parser for arithmetic expressions
with optional SI unit specifications.

Example valid inputs:
  1 [m/s]
  (1+1) [m/s^2]
  1 [kg] * 9.8 [m/s^2]
  -3.14 [mol] / (2 [cd])

Grammar (informal EBNF):

ArithmeticExpr -> Term (("+" | "-") Term)*
Term           -> Factor (("*" | "/") Factor)*
Factor         -> NumberWithOptionalUnit
                | "(" ArithmeticExpr ")"

NumberWithOptionalUnit -> NUMBER [ UnitSpec ]
UnitSpec               -> "[" UnitExpr "]"
UnitExpr               -> UnitTerm (("*" | "/") UnitTerm)*
UnitTerm               -> UnitFactor ["^" Exponent]
UnitFactor             -> "(" UnitExpr ")" | Base
Base                   -> "s" | "m" | "kg" | "A" | "K" | "mol" | "cd"
Exponent               -> INTEGER (optionally signed in the lexer)

"""

# ========== 1. LEXER ==========

def tokenize(text):
    """
    Convert the input string into a list of tokens.
    Recognized tokens:
      - Parentheses: '(' or ')'
      - Brackets for units: '[' or ']'
      - Arithmetic operators: '+', '-', '*', '/'
      - Caret: '^'
      - SI base units: s, m, kg, A, K, mol, cd
      - Numbers: integers or floats, optionally signed (e.g. -3.14, +2, 1.0, 42)
    Whitespace is ignored.
    """
    tokens = []
    i = 0
    length = len(text)

    while i < length:
        c = text[i]

        # Skip whitespace
        if c.isspace():
            i += 1
            continue

        # Single-char tokens
        if c in ('(', ')', '[', ']', '+', '-', '*', '/', '^'):
            # Distinguish minus as operator vs minus as sign for a number.
            # We'll keep '-' as its own token; if it appears directly before a digit or a decimal point,
            # we can interpret it as part of a signed number in the parser logic (or keep it separate).
            tokens.append(c)
            i += 1
            continue

        # Letters -> Could be base unit or start of 'kg'/'mol' etc.
        if c.isalpha():
            start = i
            while i < length and text[i].isalpha():
                i += 1
            token_value = text[start:i]  # e.g. "kg", "mol"
            tokens.append(token_value)
            continue

        # Digits or decimal -> parse as (possibly) a float
        if c.isdigit() or c == '.':
            start = i
            # Consume leading digits/decimal point
            i += 1
            has_dot = (c == '.')
            while i < length and (text[i].isdigit() or (text[i] == '.' and not has_dot)):
                if text[i] == '.':
                    has_dot = True
                i += 1
            token_value = text[start:i]
            tokens.append(token_value)
            continue

        # If we get here, we might be seeing a sign (+/-) before digits as part of a single token
        if (c in ('+', '-') and i + 1 < length and (text[i+1].isdigit() or text[i+1] == '.')):
            # parse sign + number
            start = i
            i += 1
            has_dot = False
            # We handle the next potential digit(s) or decimal
            if i < length and text[i] == '.':
                has_dot = True
                i += 1
            while i < length and (text[i].isdigit() or (text[i] == '.' and not has_dot)):
                if text[i] == '.':
                    has_dot = True
                i += 1
            token_value = text[start:i]
            tokens.append(token_value)
            continue

        # Otherwise, unrecognized character
        raise ValueError(f"Unrecognized character '{c}' at index {i}")

    return tokens


# ========== 2. PARSER ==========

class Parser:
    """
    Recursive-descent parser for the grammar:

      ArithmeticExpr -> Term (("+" | "-") Term)*
      Term           -> Factor (("*" | "/") Factor)*
      Factor         -> NumberWithOptionalUnit
                      | "(" ArithmeticExpr ")"

      NumberWithOptionalUnit -> NUMBER [ UnitSpec ]
      UnitSpec               -> "[" UnitExpr "]"
      UnitExpr               -> UnitTerm (("*" | "/") UnitTerm)*
      UnitTerm               -> UnitFactor ["^" Exponent]
      UnitFactor             -> "(" UnitExpr ")" | Base
      Base                   -> "s" | "m" | "kg" | "A" | "K" | "mol" | "cd"
      Exponent               -> INTEGER
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None  # End of stream

    def advance(self):
        self.pos += 1

    def match(self, *expected_values):
        """
        If the current token is in the expected_values, consume it and return it.
        Otherwise return None.
        """
        tok = self.current_token()
        if tok in expected_values:
            self.advance()
            return tok
        return None

    def expect(self, *expected_values):
        """
        If the current token is one of expected_values, consume and return it.
        Otherwise, raise an error.
        """
        tok = self.current_token()
        if tok in expected_values:
            self.advance()
            return tok
        raise ValueError(
            f"Parse error at token {tok}, expected one of {expected_values}."
        )

    def parse(self):
        """
        Parse the entire input as an ArithmeticExpr, and ensure no extra tokens.
        """
        expr = self.parse_arithmetic_expr()
        if self.current_token() is not None:
            raise ValueError(f"Extra tokens after valid expression: {self.current_token()}")
        return expr

    # ---------- Arithmetic Grammar ----------

    def parse_arithmetic_expr(self):
        """
        ArithmeticExpr -> Term (("+" | "-") Term)*
        """
        node = self.parse_term()

        while True:
            tok = self.match("+", "-")
            if not tok:
                break
            right = self.parse_term()
            node = ("binop", tok, node, right)

        return node

    def parse_term(self):
        """
        Term -> Factor (("*" | "/") Factor)*
        """
        node = self.parse_factor()

        while True:
            tok = self.match("*", "/")
            if not tok:
                break
            right = self.parse_factor()
            node = ("binop", tok, node, right)

        return node

    def parse_factor(self):
        """
        Factor -> NumberWithOptionalUnit | "(" ArithmeticExpr ")"
        """
        if self.match("("):
            # ( ArithmeticExpr )
            expr = self.parse_arithmetic_expr()
            self.expect(")")
            return expr
        else:
            return self.parse_number_with_unit()

    # ---------- Number + Optional Unit ----------

    def parse_number_with_unit(self):
        """
        NumberWithOptionalUnit -> NUMBER [ UnitSpec ]
        NUMBER can be an integer or float, optionally signed (e.g. 3, -2, 1.5).
        """
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of input while parsing number.")

        # Validate that this is indeed a numeric token
        if self._is_number(tok):
            self.advance()
            number_node = ("number", tok)
        else:
            raise ValueError(f"Expected numeric value, got '{tok}'")

        # Check if next token is '[' (start of unit spec)
        if self.match("["):
            unit_node = self.parse_unit_expr()
            self.expect("]")
            return ("measurement", number_node, unit_node)
        else:
            # Just a plain number without a unit
            return number_node

    # ---------- Unit Grammar ----------

    def parse_unit_expr(self):
        """
        UnitExpr -> UnitTerm (("*" | "/") UnitTerm)*
        """
        node = self.parse_unit_term()

        while True:
            tok = self.match("*", "/")
            if not tok:
                break
            right = self.parse_unit_term()
            node = ("unit-binop", tok, node, right)

        return node

    def parse_unit_term(self):
        """
        UnitTerm -> UnitFactor ["^" Exponent]
        """
        node = self.parse_unit_factor()

        if self.match("^"):
            exp = self.parse_exponent()
            node = ("unit-exponent", node, exp)

        return node

    def parse_unit_factor(self):
        """
        UnitFactor -> "(" UnitExpr ")" | Base
        """
        if self.match("("):
            expr = self.parse_unit_expr()
            self.expect(")")
            return expr
        else:
            return self.parse_base()

    def parse_base(self):
        """
        Base -> "s" | "m" | "kg" | "A" | "K" | "mol" | "cd"
        """
        bases = ["s", "m", "kg", "A", "K", "mol", "cd"]
        tok = self.current_token()
        if tok in bases:
            self.advance()
            return ("base", tok)
        else:
            raise ValueError(f"Expected one of {bases}, got '{tok}'")

    def parse_exponent(self):
        """
        Exponent -> INTEGER (we allow it to be a signed integer token from the lexer)
        Example: 2, -3, +4
        """
        tok = self.current_token()
        if tok is None:
            raise ValueError("Expected exponent number, found end of input.")

        # Check if it's an integer (possibly with sign)
        # We simply ensure that removing leading + or - leaves digits
        if tok.lstrip("+-").isdigit():
            self.advance()
            return ("exponent", tok)
        else:
            raise ValueError(f"Expected integer exponent, got '{tok}'")

    # ---------- Helpers ----------

    @staticmethod
    def _is_number(token):
        """
        Check if a token looks like an integer or float (including possible leading sign).
        We'll use a simple approach here: attempt float() conversion.
        """
        try:
            float(token)
            return True
        except ValueError:
            return False


# ========== 3. TEST / MAIN ==========

def main():
    # Try different expressions here:
    TEST_INPUTS = [
        "1 [m/s]",
        "(1+1) [m/s^2]",
        "1 [kg] * 9.8 [m/s^2]",
        "-3.14 [mol] / (2 [cd])",
        "((2+2)*3) [s]",
        "2.5 [kg*m/(s^2)] - 4 [A^2]"
    ]

    for inp in TEST_INPUTS:
        print(f"\nInput: {inp}")
        try:
            tokens = tokenize(inp)
            parser = Parser(tokens)
            parse_tree = parser.parse()
            print("Tokens:", tokens)
            print("Parse Tree:", parse_tree)
        except ValueError as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
