import ast
import unittest
from pathlib import Path
from tomllib import loads

import config

ROOT = Path(__file__).resolve().parents[1]


class PublicContractTest(unittest.TestCase):
    def test_package_and_openapi_versions_match(self) -> None:
        package_version = loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        tree = ast.parse((ROOT / "main.py").read_text())
        openapi_version = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
                for keyword in node.keywords:
                    if keyword.arg == "version" and isinstance(keyword.value, ast.Constant):
                        openapi_version = keyword.value.value

        self.assertEqual(package_version, openapi_version)

    def test_documented_input_limit_matches_runtime_default(self) -> None:
        readme = (ROOT / "README.md").read_text()
        self.assertIn(f"| `EMBEDDING_MAX_LENGTH` | `{config.MAX_INPUT_LENGTH}` |", readme)

    def test_readme_lists_every_live_operational_endpoint(self) -> None:
        readme = (ROOT / "README.md").read_text()
        for endpoint in ("/health", "/metrics", "/v1/warmup"):
            self.assertIn(endpoint, readme)


if __name__ == "__main__":
    unittest.main()
