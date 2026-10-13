"""Resolve deterministic relationships between repository files."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from cohere_workshops.mapper import RepositoryMap


@dataclass(frozen=True)
class ImportRelationship:
    """A resolved local Python import edge."""

    source_path: str
    target_path: str
    import_name: str


def _python_module_names(path: str) -> tuple[str, ...]:
    """Return importable module names represented by a Python path."""
    if not path.endswith(".py"):
        return ()

    parts = path.removesuffix(".py").split("/")

    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        return ()

    names = [".".join(parts)]

    if parts[0] == "src" and len(parts) > 1:
        names.append(".".join(parts[1:]))

    return tuple(names)


def resolve_import_relationships(
    repository_map: RepositoryMap,
) -> tuple[ImportRelationship, ...]:
    """Resolve local imports without inferring external dependencies."""
    module_paths: dict[str, set[str]] = defaultdict(set)

    for mapped_file in repository_map.files:
        for module_name in _python_module_names(mapped_file.path):
            module_paths[module_name].add(mapped_file.path)

    relationships: set[ImportRelationship] = set()

    for mapped_file in repository_map.files:
        if mapped_file.language != "python":
            continue

        for import_name in mapped_file.imports:
            for target_path in module_paths.get(import_name, set()):
                if target_path != mapped_file.path:
                    relationships.add(
                        ImportRelationship(
                            source_path=mapped_file.path,
                            target_path=target_path,
                            import_name=import_name,
                        )
                    )

    return tuple(
        sorted(
            relationships,
            key=lambda item: (
                item.source_path,
                item.target_path,
                item.import_name,
            ),
        )
    )


def related_paths(
    seed_paths: Iterable[str],
    relationships: Iterable[ImportRelationship],
) -> tuple[str, ...]:
    """Return deterministic one-hop neighbors of seed files."""
    seeds = set(seed_paths)
    related: set[str] = set()

    for relationship in relationships:
        if relationship.source_path in seeds:
            related.add(relationship.target_path)
        if relationship.target_path in seeds:
            related.add(relationship.source_path)

    return tuple(sorted(related - seeds))
