#!/usr/bin/env python3

##############################################
# 1. DIMENSION CONSTANTS & HELPERS
##############################################

# We store dimension vectors in the order [L, M, T, I, Θ, N, J].
#   L = length        (m)
#   M = mass          (kg)
#   T = time          (s)
#   I = current       (A)
#   Θ = temperature   (K)
#   N = amount        (mol)
#   J = luminous int. (cd)

BASE_DIMENSIONS = {
    "m":   [1, 0, 0, 0, 0, 0, 0],
    "kg":  [0, 1, 0, 0, 0, 0, 0],
    "s":   [0, 0, 1, 0, 0, 0, 0],
    "A":   [0, 0, 0, 1, 0, 0, 0],
    "K":   [0, 0, 0, 0, 1, 0, 0],
    "mol": [0, 0, 0, 0, 0, 1, 0],
    "cd":  [0, 0, 0, 0, 0, 0, 1],
}

def zero_dim():
    return [0, 0, 0, 0, 0, 0, 0]

def dim_add(d1, d2):
    return [a + b for (a, b) in zip(d1, d2)]

def dim_sub(d1, d2):
    return [a - b for (a, b) in zip(d1, d2)]

def dim_mul(d, n):
    return [x * n for x in d]

def dim_eq(d1, d2):
    return all(a == b for a, b in zip(d1, d2))

##############################################
# 2. AST NODES
##############################################

class NumberNode:
    """
    Numeric literal (string) + optional raw unit string from brackets.
    E.g. value_str="1.5", unit_str="kg" or "m/s^2" or None if dimensionless.
    """
    def __init__(self, value_str, unit_str=None):
        self.value_str = value_str
        self.unit_str = unit_str

    def __repr__(self):
        if self.unit_str:
            return f"NumberNode(value={self.value_str}, unit=[{self.unit_str}])"
        return f"NumberNode(value={self.value_str})"


class BinOpNode:
    """
    Operator node with left, op, right.
    op in {+ - * / ^}
    """
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinOpNode({self.left} {self.op} {self.right})"

##############################################
# 3. LEXER
##############################################

def tokenize(text):
    """
    Splits the input into tokens:
      - numeric tokens (possibly signed)
      - parentheses: ( )
      - operators: + - * / ^
      - bracketed unit chunk: anything from '[' to ']' is one token, e.g. [m/s^2]
      - whitespace ignored
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if c.isspace():
            i += 1
            continue

        # Parentheses or operators
        if c in ('(', ')', '+', '*', '/', '^'):
            tokens.append(c)
            i += 1
            continue

        # Distinguish minus as unary sign vs. operator
        if c == '-':
            # If next char is digit or '.', treat as part of a number
            if (i + 1 < n) and (text[i+1].isdigit() or text[i+1] == '.'):
                start = i
                i += 1
                while i < n and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                tokens.append(text[start:i])  # e.g. "-3.14"
            else:
                tokens.append('-')
                i += 1
            continue

        # Bracketed unit: read everything up to matching ']'
        if c == '[':
            start = i
            i += 1
            # find the matching ']'
            bracket_depth = 1
            while i < n and bracket_depth > 0:
                if text[i] == ']':
                    bracket_depth -= 1
                else:
                    i += 1
            if bracket_depth != 0:
                raise ValueError("Mismatched '[' ']' in unit specification.")
            # i now points to the ']' => i += 1 to include it
            i += 1
            # The entire bracketed substring
            tokens.append(text[start:i])  # e.g. "[m/s^2]"
            continue

        # Numbers (unsigned, if minus was handled above)
        if c.isdigit() or c == '.':
            start = i
            i += 1
            while i < n and (text[i].isdigit() or text[i] == '.'):
                i += 1
            tokens.append(text[start:i])
            continue

        raise ValueError(f"Unexpected character '{c}' at index {i}")

    return tokens

##############################################
# 4. PARSER
##############################################

class Parser:
    """
    Grammar:
      Expr   -> Term (('+'|'-') Term)*
      Term   -> Factor (('*'|'/') Factor)*
      Factor -> Power ('^' Factor)?   // exponent
      Power  -> NumberWithOptionalUnit | '(' Expr ')'
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def match(self, *expected):
        tok = self.current_token()
        if tok in expected:
            self.pos += 1
            return tok
        return None

    def expect(self, *expected):
        tok = self.current_token()
        if tok not in expected:
            raise ValueError(f"Expected one of {expected}, got {tok}")
        self.pos += 1
        return tok

    def parse(self):
        node = self.parse_expr()
        if self.current_token() is not None:
            raise ValueError(f"Extra tokens after valid expression: {self.current_token()}")
        return node

    def parse_expr(self):
        """Expr -> Term (('+'|'-') Term)*"""
        node = self.parse_term()
        while True:
            op = self.match('+', '-')
            if not op:
                break
            right = self.parse_term()
            node = BinOpNode(node, op, right)
        return node

    def parse_term(self):
        """Term -> Factor (('*'|'/') Factor)*"""
        node = self.parse_factor()
        while True:
            op = self.match('*', '/')
            if not op:
                break
            right = self.parse_factor()
            node = BinOpNode(node, op, right)
        return node

    def parse_factor(self):
        """Factor -> Power ('^' Factor)?"""
        base = self.parse_power()
        if self.match('^'):
            exponent = self.parse_factor()
            return BinOpNode(base, '^', exponent)
        return base

    def parse_power(self):
        """Power -> NumberWithOptionalUnit | '(' Expr ')'"""
        if self.match('('):
            node = self.parse_expr()
            self.expect(')')
            return node
        else:
            return self.parse_number_with_unit()

    def parse_number_with_unit(self):
        """
        A numeric token plus optional bracket token.
        e.g.  "3.14" or "3.14 [m/s^2]"
        """
        tok = self.current_token()
        if tok is None:
            raise ValueError("Expected number, got end of input.")
        # Validate numeric
        # If this fails, it's not a number
        try:
            float(tok)
        except ValueError:
            raise ValueError(f"Expected numeric token, got {tok}")
        self.pos += 1  # consume the numeric

        # Check if next token is a bracketed unit
        unit_tok = None
        if self.current_token() and self.current_token().startswith('['):
            unit_tok = self.current_token()
            self.pos += 1  # consume it

        return NumberNode(value_str=tok, unit_str=unit_tok)

##############################################
# 5. INTERPRETER (WITH DIMENSIONAL ANALYSIS)
##############################################

def evaluate(node):
    """Recursively evaluate the AST node, returning (value: float, dim: list[7])."""
    if isinstance(node, NumberNode):
        val = float(node.value_str)
        # Parse unit string (if any) into a dimension vector
        if not node.unit_str:
            dim = zero_dim()  # dimensionless
        else:
            dim = parse_unit_string(node.unit_str)
        return (val, dim)

    elif isinstance(node, BinOpNode):
        left_val, left_dim = evaluate(node.left)
        right_val, right_dim = evaluate(node.right)
        op = node.op

        if op == '+':
            if not dim_eq(left_dim, right_dim):
                raise ValueError(
                    f"Dimension mismatch in addition: {left_dim} vs {right_dim}"
                )
            return (left_val + right_val, left_dim)

        elif op == '-':
            if not dim_eq(left_dim, right_dim):
                raise ValueError(
                    f"Dimension mismatch in subtraction: {left_dim} vs {right_dim}"
                )
            return (left_val - right_val, left_dim)

        elif op == '*':
            return (left_val * right_val, dim_add(left_dim, right_dim))

        elif op == '/':
            if right_val == 0:
                raise ValueError("Division by zero in numeric part.")
            return (left_val / right_val, dim_sub(left_dim, right_dim))

        elif op == '^':
            # exponent must be dimensionless & typically an integer
            if not dim_eq(right_dim, zero_dim()):
                raise ValueError("Exponent must be dimensionless.")
            exponent_int = int(right_val)  # allow float->int
            return (left_val ** exponent_int, dim_mul(left_dim, exponent_int))

        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        raise TypeError("Invalid AST node type.")

def parse_unit_string(unit_tok):
    """
    Convert a bracketed unit token (e.g. "[m/s^2]" or "[-]") into a dimension vector.
    We strip off the brackets first, then parse what's inside.
    """
    # e.g. if unit_tok="[m/s^2]", inside="m/s^2"
    inside = unit_tok.strip()
    if not inside.startswith('[') or not inside.endswith(']'):
        raise ValueError(f"Bad bracket token: {unit_tok}")

    inside = inside[1:-1].strip()  # remove '[' and ']'

    # Special case: "[-]" means dimensionless
    if inside == '-' or inside == '':
        return zero_dim()

    # We'll do a very simple parse: split on '*' and '/' in order.
    # e.g. "m/s^2" => "m", "/", "s^2"
    import re
    parts = re.split(r'([\*/])', inside)
    parts = [p.strip() for p in parts if p.strip()]

    dim = zero_dim()
    current_op = '*'

    for part in parts:
        if part in ('*', '/'):
            current_op = part
        else:
            # part might be "m", "s^2", "kg^3"
            base, exponent = parse_base_exponent(part)
            if base not in BASE_DIMENSIONS:
                raise ValueError(
                    f"Unknown base unit '{base}'. Must be one of {list(BASE_DIMENSIONS.keys())}, or '[-]' for dimensionless."
                )
            factor_dim = dim_mul(BASE_DIMENSIONS[base], exponent)

            if current_op == '*':
                dim = dim_add(dim, factor_dim)
            else:  # '/'
                dim = dim_sub(dim, factor_dim)

    return dim

def parse_base_exponent(token):
    """
    For something like "m^2" => (base="m", exponent=2).
    If there's no '^', exponent=1.
    """
    if '^' in token:
        base, exp_str = token.split('^', 1)
        e = int(exp_str.strip())
    else:
        base = token
        e = 1
    return base.strip(), e

##############################################
# 6. DEMO
##############################################

def main():
    examples = [
        "2 [m] + 3 [m]",
        "2 [m] + 3 [s]",         # dimension mismatch
        "1.5 [kg] * 2 [m/s^2]",
        "10 [m] / 5 [s]",
        "(2 [m/s])^2",
        "3 [m] - 1 [m]",
        "3 [m] - 1 [s]",
        "3.14 [-]",             # dimensionless
        "-2 [s] * 5 [m]",
        "2.0 [kg] / 2 [kg]"     # dimensionless result
    ]

    for expr in examples:
        print(f"\nEXPRESSION: {expr}")
        try:
            tokens = tokenize(expr)
            parser = Parser(tokens)
            ast = parser.parse()
            (value, dim) = evaluate(ast)
            print("  => Tokens:", tokens)
            print("  => AST:", ast)
            print(f"  => Value: {value}")
            print(f"  => Dimension [L,M,T,I,Θ,N,J]: {dim}")
        except Exception as e:
            print("  ERROR:", e)

if __name__ == "__main__":
    main()
