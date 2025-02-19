#!/usr/bin/env python3

"""
A simple lexer and parser for expressions of the form:
  Expression -> Term { ("*" | "/") Term }
  Term       -> Factor [ "^" Exponent ]
  Factor     -> "(" Expression ")" | Base
  Base       -> s | m | kg | A | K | mol | cd
  Exponent   -> integer (optionally with + or - sign)

Examples of valid inputs:
  m
  kg * m / s^2
  (m * kg) / (s^2 * mol)
  A^3

Usage:
  python si_parser.py
  # Then type or modify the 'TEST_INPUT' string in main()
"""

# ========== 1. LEXER ==========

def tokenize(text):
    """
    Convert the input string into a list of tokens.
    Tokens include:
      - SI base units: 's', 'm', 'kg', 'A', 'K', 'mol', 'cd'
      - Operators: '*', '/', '^'
      - Parentheses: '(', ')'
      - Integers (optionally with leading + or -)
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
        if c in ('*', '/', '(', ')', '^'):
            tokens.append(c)
            i += 1
            continue

        # Potentially multi-letter base units or sign+number
        if c.isalpha():
            # Collect consecutive letters (e.g. 'kg', 'mol')
            start = i
            while i < length and text[i].isalpha():
                i += 1
            token_value = text[start:i]
            tokens.append(token_value)
            continue

        # Integers (with optional + or - prefix)
        if c.isdigit() or c in ('+', '-'):
            start = i
            # If there's a leading sign, consume it
            if c in ('+', '-'):
                i += 1
                # Check if the next character is digit
                if i >= length or not text[i].isdigit():
                    raise ValueError(f"Invalid numeric token starting at '{text[start:]}'")
            # Now consume all digits
            while i < length and text[i].isdigit():
                i += 1
            token_value = text[start:i]
            tokens.append(token_value)
            continue

        # If we get here, it's an unexpected character
        raise ValueError(f"Unexpected character in input: '{c}' at index {i}")

    return tokens


# ========== 2. PARSER ==========

class Parser:
    """
    A simple recursive-descent parser based on the grammar:

      Expression -> Term { ("*" | "/") Term }
      Term       -> Factor [ "^" Exponent ]
      Factor     -> "(" Expression ")" | Base
      Base       -> s | m | kg | A | K | mol | cd
      Exponent   -> integer (with optional + or -)
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        self.pos += 1

    def match(self, expected):
        """
        If the current token matches 'expected', consume it and return True.
        Otherwise return False.
        """
        if self.current_token() == expected:
            self.advance()
            return True
        return False

    def expect(self, expected):
        """
        Consume the current token if it matches 'expected'.
        Otherwise, raise a parsing error.
        """
        if not self.match(expected):
            raise ValueError(f"Parsing error: expected '{expected}', got '{self.current_token()}'")

    # ----- Grammar rules -----

    def parse(self):
        """
        Entry point: parse an Expression and check for extra tokens.
        Returns a parse tree (nested tuples).
        """
        node = self.parse_expression()
        # If there are leftover tokens, it's an error
        if self.current_token() is not None:
            raise ValueError(f"Extra tokens after valid expression: {self.current_token()}")
        return node

    def parse_expression(self):
        """
        Expression -> Term { ("*" | "/") Term }
        """
        node = self.parse_term()

        while self.current_token() in ("*", "/"):
            op = self.current_token()
            self.advance()  # consume '*' or '/'
            right = self.parse_term()
            node = ("binop", op, node, right)

        return node

    def parse_term(self):
        """
        Term -> Factor [ "^" Exponent ]
        """
        node = self.parse_factor()

        if self.match("^"):
            # We expect an integer token (optionally with leading + or -).
            exponent_token = self.current_token()
            if exponent_token is None:
                raise ValueError("Parsing error: missing exponent after '^'")

            # Validate it's numeric
            if exponent_token.lstrip('+-').isdigit():
                self.advance()
                node = ("exponent", node, exponent_token)
            else:
                raise ValueError(f"Parsing error: invalid exponent token '{exponent_token}'")

        return node

    def parse_factor(self):
        """
        Factor -> "(" Expression ")" | Base
        """
        tok = self.current_token()
        if tok == "(":
            self.advance()
            node = self.parse_expression()
            self.expect(")")
            return node
        else:
            return self.parse_base()

    def parse_base(self):
        """
        Base -> s | m | kg | A | K | mol | cd
        """
        tok = self.current_token()
        valid_bases = ["s", "m", "kg", "A", "K", "mol", "cd"]
        if tok in valid_bases:
            self.advance()
            return ("base", tok)
        else:
            raise ValueError(f"Parsing error: expected a base unit, got '{tok}'")


# ========== 3. TESTING OR MAIN USAGE ==========

def main():
    # Modify the input string to test different expressions
    TEST_INPUT = "kg*m / s^2"

    try:
        # Lex
        tokens = tokenize(TEST_INPUT)
        print("Tokens:", tokens)

        # Parse
        parser = Parser(tokens)
        parse_tree = parser.parse()

        # Print parse tree
        print("Parse tree:", parse_tree)

    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
