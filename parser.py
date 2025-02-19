#!/usr/bin/env python3

"""
A parser for arithmetic expressions with SI units, producing an AST
where each node stores:
  (a) a floating-point value
  (b) a 7-element dimension vector for the SI base units:
       (time, length, mass, current, temperature, amount, luminous intensity)

Example inputs:
  1 [m/s]
  (1+1) [m/s^2]
  1 [kg] * 9.8 [m/s^2]
  2 [m] + 3 [m]   # valid
  2 [m] + 3 [s]   # dimension mismatch -> error
  5 [kg] / 2 [s^2]
  -3 [mol] / (2 [cd])
"""

# ----------------------------------------------
# 1. Dimension Vector Utilities
# ----------------------------------------------

# We'll store dimension vectors as tuples/lists of length 7:
#    [ time, length, mass, current, temperature, amount, luminosity ]
#
# Example base units:
#   s   -> [ 1, 0, 0, 0, 0, 0, 0 ]
#   m   -> [ 0, 1, 0, 0, 0, 0, 0 ]
#   kg  -> [ 0, 0, 1, 0, 0, 0, 0 ]
#   A   -> [ 0, 0, 0, 1, 0, 0, 0 ]
#   K   -> [ 0, 0, 0, 0, 1, 0, 0 ]
#   mol -> [ 0, 0, 0, 0, 0, 1, 0 ]
#   cd  -> [ 0, 0, 0, 0, 0, 0, 1 ]

BASE_DIMENSIONS = {
    "s":   [1, 0, 0, 0, 0, 0, 0],
    "m":   [0, 1, 0, 0, 0, 0, 0],
    "kg":  [0, 0, 1, 0, 0, 0, 0],
    "A":   [0, 0, 0, 1, 0, 0, 0],
    "K":   [0, 0, 0, 0, 1, 0, 0],
    "mol": [0, 0, 0, 0, 0, 1, 0],
    "cd":  [0, 0, 0, 0, 0, 0, 1],
}


def dim_add(d1, d2):
    """Add two dimension vectors (for multiplication)."""
    return [a + b for (a, b) in zip(d1, d2)]

def dim_sub(d1, d2):
    """Subtract two dimension vectors (for division)."""
    return [a - b for (a, b) in zip(d1, d2)]

def dim_mul(d, factor):
    """Multiply all elements of a dimension vector by an integer exponent."""
    return [x * factor for x in d]

def dim_eq(d1, d2):
    """Check if two dimension vectors are equal."""
    return all(a == b for (a, b) in zip(d1, d2))

def zero_dim():
    """Return a zero dimension vector (dimensionless)."""
    return [0, 0, 0, 0, 0, 0, 0]


# ----------------------------------------------
# 2. AST Node Definition
# ----------------------------------------------

class ASTNode:
    """
    Represents a node in the final abstract syntax tree, storing:
      - numeric_value: float
      - dimension: a 7-element list of exponents
      - node_type: "number", "binop", etc.
      - operator: for binop nodes, the operator symbol
      - left, right: children (for binop)
    """

    def __init__(self, node_type, numeric_value, dimension, operator=None, left=None, right=None):
        self.node_type = node_type  # "number" or "binop"
        self.numeric_value = numeric_value  # float
        self.dimension = dimension         # 7-element list
        self.operator = operator           # e.g. "+", "-", "*", "/"
        self.left = left                   # ASTNode
        self.right = right                 # ASTNode

    def __repr__(self):
        if self.node_type == "number":
            return f"<AST number={self.numeric_value}, dim={self.dimension}>"
        elif self.node_type == "binop":
            return (f"<AST binop='{self.operator}' dim={self.dimension}>\n"
                    f"  left={self.left}\n"
                    f"  right={self.right}>\n")
        else:
            return f"<AST unknown type={self.node_type}>"


# ----------------------------------------------
# 3. LEXER
# ----------------------------------------------

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

        # Single-char tokens (or easily recognized)
        if c in ('(', ')', '[', ']', '+', '*', '/', '^'):
            tokens.append(c)
            i += 1
            continue

        # Distinguish minus as operator vs. minus as part of a number
        # Strategy: If '-' is followed by a digit or a dot, treat it as signed number
        if c == '-':
            if i + 1 < length and (text[i+1].isdigit() or text[i+1] == '.'):
                # It's a negative number
                start = i
                i += 1
                # consume digits or decimal
                has_dot = False
                while i < length and (text[i].isdigit() or (text[i] == '.' and not has_dot)):
                    if text[i] == '.':
                        has_dot = True
                    i += 1
                tokens.append(text[start:i])
            else:
                # It's just the minus operator
                tokens.append('-')
                i += 1
            continue

        # Letters -> Could be base unit, e.g. 'kg', 'mol'
        if c.isalpha():
            start = i
            while i < length and text[i].isalpha():
                i += 1
            tokens.append(text[start:i])  # e.g. "kg", "mol"
            continue

        # Digits or decimal -> parse as (possibly) a float
        if c.isdigit() or c == '.':
            start = i
            has_dot = (c == '.')
            i += 1
            while i < length and (text[i].isdigit() or (text[i] == '.' and not has_dot)):
                if text[i] == '.':
                    has_dot = True
                i += 1
            tokens.append(text[start:i])
            continue

        raise ValueError(f"Unrecognized character '{c}' at index {i}")

    return tokens


# ----------------------------------------------
# 4. PARSER
# ----------------------------------------------

class Parser:
    """
    A recursive-descent parser that constructs an AST where each node
    includes a numeric value and a 7-element dimension vector.
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
        If the current token is in expected_values, consume and return it.
        Otherwise return None.
        """
        tok = self.current_token()
        if tok in expected_values:
            self.advance()
            return tok
        return None

    def expect(self, *expected_values):
        tok = self.current_token()
        if tok in expected_values:
            self.advance()
            return tok
        raise ValueError(f"Parse error: expected one of {expected_values}, got '{tok}'")

    def parse(self):
        """
        Parse the entire input as an arithmetic expression.
        Return the root ASTNode.
        """
        node = self.parse_arithmetic_expr()
        if self.current_token() is not None:
            raise ValueError(f"Extra tokens after a valid expression: {self.current_token()}")
        return node

    # --------------------------------------
    # 4.1 Arithmetic Grammar
    # --------------------------------------

    def parse_arithmetic_expr(self):
        """
        ArithmeticExpr -> Term (("+" | "-") Term)*
        """
        node = self.parse_term()

        while True:
            op = self.match("+", "-")
            if not op:
                break
            right = self.parse_term()
            node = self.make_binop_node(op, node, right)
        return node

    def parse_term(self):
        """
        Term -> Factor (("*" | "/") Factor)*
        """
        node = self.parse_factor()

        while True:
            op = self.match("*", "/")
            if not op:
                break
            right = self.parse_factor()
            node = self.make_binop_node(op, node, right)
        return node

    def parse_factor(self):
        """
        Factor -> NumberWithOptionalUnit | "(" ArithmeticExpr ")"
        """
        if self.match("("):
            expr_node = self.parse_arithmetic_expr()
            self.expect(")")
            return expr_node
        else:
            return self.parse_number_with_optional_unit()

    # --------------------------------------
    # 4.2 Number + Optional Unit
    # --------------------------------------

    def parse_number_with_optional_unit(self):
        """
        Parse a numeric token (integer or float, possibly negative)
        followed by an optional unit specification in brackets:
          e.g. 3.14 [m/s]
        """
        tok = self.current_token()
        if tok is None:
            raise ValueError("Expected number, found end of input.")

        # Parse the numeric portion
        num_value = self.parse_float(tok)
        self.advance()

        dim = zero_dim()  # dimensionless by default

        # Check if next token is '[' -> unit specification
        if self.match("["):
            # parse unit expression, then expect ']'
            dim = self.parse_unit_expr()
            self.expect("]")

        # Return an AST node with the numeric value + dimension
        return ASTNode(
            node_type="number",
            numeric_value=num_value,
            dimension=dim
        )

    @staticmethod
    def parse_float(token):
        """Try converting token to float."""
        try:
            return float(token)
        except ValueError:
            raise ValueError(f"Expected a numeric token, got '{token}'")

    # --------------------------------------
    # 4.3 Unit Grammar
    # --------------------------------------

    def parse_unit_expr(self):
        """
        UnitExpr -> UnitTerm (("*" | "/") UnitTerm)*
        Returns a dimension vector.
        """
        dim = self.parse_unit_term()
        while True:
            op = self.match("*", "/")
            if not op:
                break
            right_dim = self.parse_unit_term()
            if op == "*":
                dim = dim_add(dim, right_dim)
            else:  # op == "/"
                dim = dim_sub(dim, right_dim)
        return dim

    def parse_unit_term(self):
        """
        UnitTerm -> UnitFactor ["^" Exponent]
        Returns a dimension vector.
        """
        dim = self.parse_unit_factor()
        if self.match("^"):
            exponent_token = self.current_token()
            if exponent_token is None:
                raise ValueError("Expected exponent after '^'")
            # Check exponent is integer
            if exponent_token.lstrip('+-').isdigit():
                exponent_val = int(exponent_token)
                self.advance()
                dim = dim_mul(dim, exponent_val)
            else:
                raise ValueError(f"Invalid exponent '{exponent_token}'")
        return dim

    def parse_unit_factor(self):
        """
        UnitFactor -> "(" UnitExpr ")" | Base
        Returns a dimension vector.
        """
        if self.match("("):
            dim = self.parse_unit_expr()
            self.expect(")")
            return dim
        else:
            return self.parse_base()

    def parse_base(self):
        """
        Base -> s | m | kg | A | K | mol | cd
        Returns a dimension vector for that base.
        """
        tok = self.current_token()
        if tok in BASE_DIMENSIONS:
            self.advance()
            return BASE_DIMENSIONS[tok][:]
        else:
            raise ValueError(f"Expected base unit, got '{tok}'")

    # --------------------------------------
    # 4.4 Building BinOp Nodes
    # --------------------------------------

    def make_binop_node(self, op, left_node, right_node):
        """
        Build a binop ASTNode given operator (op), left_node, right_node.
        We also compute the dimension vector and numeric_value.
        """
        # 1) Dimension logic
        if op in ("+", "-"):
            # dimension vectors must match
            if not dim_eq(left_node.dimension, right_node.dimension):
                raise ValueError(
                    f"Dimension mismatch in '{op}' operation: "
                    f"{left_node.dimension} vs {right_node.dimension}"
                )
            dim_result = left_node.dimension[:]  # same dimension
            # 2) Numeric result
            if op == "+":
                val_result = left_node.numeric_value + right_node.numeric_value
            else:  # "-"
                val_result = left_node.numeric_value - right_node.numeric_value

        elif op == "*":
            # dimension = sum
            dim_result = dim_add(left_node.dimension, right_node.dimension)
            # numeric = multiply
            val_result = left_node.numeric_value * right_node.numeric_value

        elif op == "/":
            # dimension = difference
            dim_result = dim_sub(left_node.dimension, right_node.dimension)
            # numeric = divide
            if right_node.numeric_value == 0:
                raise ValueError("Division by zero in numeric part.")
            val_result = left_node.numeric_value / right_node.numeric_value

        else:
            raise ValueError(f"Unknown operator '{op}'")

        return ASTNode(
            node_type="binop",
            numeric_value=val_result,
            dimension=dim_result,
            operator=op,
            left=left_node,
            right=right_node
        )


# ----------------------------------------------
# 5. DEMO / MAIN
# ----------------------------------------------

def main():
    TEST_INPUTS = [
        "1 [m/s]",
        "(1+1) [m/s^2]",
        "1 [kg] * 9.8 [m/s^2]",
        "2 [m] + 3 [m]",
        "2 [m] + 3 [s]",    # dimension mismatch
        "5 [kg] / 2 [s^2]",
        "-3 [mol] / (2 [cd])",
        "2 [m^1] + 5 [m]"   # same dimension
    ]

    for inp in TEST_INPUTS:
        print(f"\nExpression: {inp}")
        try:
            tokens = tokenize(inp)
            parser = Parser(tokens)
            ast_root = parser.parse()
            print("  => Tokens:", tokens)
            print("  => AST:", ast_root)
            print(f"     numeric_value = {ast_root.numeric_value}")
            print(f"     dimension_vector = {ast_root.dimension}")
        except ValueError as e:
            print("  ERROR:", e)


if __name__ == "__main__":
    main()
