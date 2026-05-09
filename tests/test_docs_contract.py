"""Regression tests for the docs directory contract."""

from pathlib import Path


DOCS_DIR = Path("docs")


def test_docs_required_files_exist() -> None:
    """Required active docs files should exist."""
    for name in ["README.md", "plan.md", "report.md", "progress.md", "issues.md"]:
        assert (DOCS_DIR / name).is_file(), f"Missing docs/{name}"


def test_docs_required_directories_exist() -> None:
    """Required docs subdirectories should exist."""
    for name in ["inbox", "references", "archive"]:
        assert (DOCS_DIR / name).is_dir(), f"Missing docs/{name}/"


def test_docs_readme_defines_active_and_archive_contract() -> None:
    """README should define active plan, inbox, archive, and reference boundaries."""
    readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")

    assert "`inbox/README.md`" in readme
    assert "`archive/README.md`" in readme
    assert "references/context_snapshot_v4_2.md" in readme


def test_docs_root_has_no_backup_markdown_files() -> None:
    """Backup markdown files should be kept under docs/archive, not docs root."""
    backup_files = [
        path
        for path in DOCS_DIR.glob("*.md")
        if "backup" in path.name.lower()
    ]

    assert backup_files == []
