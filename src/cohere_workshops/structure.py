"""Extract structural evidence from Python source files."""

import ast
from dataclasses import dataclass

from cohere_workshops.documents import RepositoryDocument


@dataclass(frozen=True)
class CodeSymbol:
    """A named symbol found in source code."""

    name: str
    qualified_name: str
    kind: str
    line: int
    end_line: int


@dataclass(frozen=True)
class PythonStructure:
    """Structural facts extracted from one Python file."""

    path: str
    symbols: tuple[CodeSymbol, ...]
    imports: tuple[str, ...]


class _SymbolVisitor(ast.NodeVisitor):
    """Collect symbols while retaining their lexical ownership."""

    def __init__(self) -> None:
        self.symbols: list[CodeSymbol] = []
        self.scope: list[tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = (
            "method"
            if self.scope and self.scope[-1][1] == "class"
            else "function"
        )
        self._visit_symbol(node, kind)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = (
            "method"
            if self.scope and self.scope[-1][1] == "class"
            else "function"
        )
        self._visit_symbol(node, kind)

    def _visit_symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        qualified_name = ".".join(
            [*(name for name, _ in self.scope), node.name]
        )
        self.symbols.append(
            CodeSymbol(
                name=node.name,
                qualified_name=qualified_name,
                kind=kind,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            )
        )
        self.scope.append((node.name, kind))
        self.generic_visit(node)
        self.scope.pop()


def analyze_python(document: RepositoryDocument) -> PythonStructure:
    """Extract symbols and imports without using a language model."""
    if document.language != "python":
        raise ValueError(f"Expected a Python document, received {document.language}")

    tree = ast.parse(document.content, filename=document.path)
    visitor = _SymbolVisitor()
    visitor.visit(tree)

    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    return PythonStructure(
        path=document.path,
        symbols=tuple(
            sorted(visitor.symbols, key=lambda symbol: symbol.line)
        ),
        imports=tuple(sorted(name for name in imports if name)),
    )
