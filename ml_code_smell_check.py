"""
This module defines a custom Pylint checker for detecting common code smells in machine learning code, particularly with Pandas and NumPy usage.
It checks for:
1. Unnecessary iteration over Pandas DataFrame rows.
2. Chain indexing in Pandas.
3. Misuse of DataFrame to ndarray conversion.
4. Misuse of matrix multiplication functions in NumPy.
"""

from pylint.checkers import BaseChecker
import astroid

class MLCodeSmellChecker(BaseChecker):

    name = "ml-code-smell-checker"
    msgs = {
        "C9001": (
            "Avoid unnecessary iteration over Pandas DataFrame rows. Use vectorized operations.",
            "unnecessary-iteration",
            "Use vectorized Pandas operations instead of row-wise iteration.",
        ),
        "C9002": (
            "Avoid chain indexing in Pandas (df[col][row]). Use .loc or .iloc instead.",
            "chain-indexing",
            "Chained indexing can cause unpredictable behavior.",
        ),
        "C9003": (
            "Use df.to_numpy() instead of df.values for DataFrame to ndarray conversion.",
            "dataframe-conversion-misused",
            "df.to_numpy() is the preferred method over df.values.",
        ),
        "C9004": (
            "Use np.matmul() for matrix multiplication instead of np.dot() for clarity.",
            "matrix-multiplication-misused",
            "np.matmul() is preferred for matrix multiplication.",
        ),
    }

    def visit_for(self, node: astroid.nodes.For) -> None:
        # Detect unnecessary iteration over DataFrame rows, e.g., for _, row in df.iterrows()
        iter_call = node.iter
        if (
            isinstance(iter_call, astroid.Call)
            and getattr(iter_call.func, "attrname", "") == "iterrows"
        ):
            self.add_message("unnecessary-iteration", node=node)

    def visit_subscript(self, node: astroid.nodes.Subscript) -> None:
        # Detect chained indexing: df[col][row] or df[col][mask]
        value = node.value
        if isinstance(value, astroid.Subscript):
            self.add_message("chain-indexing", node=node)

    def visit_attribute(self, node: astroid.nodes.Attribute) -> None:
        # Detect df.values usage
        if node.attrname == "values":
            expr = node.expr
            # Quick heuristic: check if expr looks like a dataframe variable
            self.add_message("dataframe-conversion-misused", node=node)
        
    def visit_call(self, node: astroid.nodes.Call) -> None:
        # Detect np.dot usage for matrix multiplication
        func = node.func
        if isinstance(func, astroid.Attribute):
            if func.attrname == "dot":
                inferred = next(func.expr.infer(), None)
                if inferred and inferred.qname().startswith("numpy"):
                    self.add_message("matrix-multiplication-misused", node=node)

def register(linter):
    linter.register_checker(MLCodeSmellChecker(linter))
