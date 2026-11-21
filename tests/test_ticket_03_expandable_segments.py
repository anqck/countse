import ast
import os
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch


class TestConfigureCudaAllocConf(unittest.TestCase):
    def test_linux_sets_pytorch_alloc_conf(self):
        from util.cuda_alloc_conf import configure_cuda_alloc_conf

        env = {}
        configure_cuda_alloc_conf(env=env, platform="linux")
        self.assertEqual(
            env,
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )

    def test_non_linux_platforms_are_noop(self):
        from util.cuda_alloc_conf import configure_cuda_alloc_conf

        for plat in ("win32", "darwin", "cygwin"):
            with self.subTest(platform=plat):
                env = {}
                configure_cuda_alloc_conf(env=env, platform=plat)
                self.assertEqual(env, {})

                preexisting = {"PYTORCH_ALLOC_CONF": "max_split_size_mb:64"}
                configure_cuda_alloc_conf(env=preexisting, platform=plat)
                self.assertEqual(
                    preexisting,
                    {"PYTORCH_ALLOC_CONF": "max_split_size_mb:64"},
                )

        with patch.object(sys, "platform", "win32"):
            env = {}
            configure_cuda_alloc_conf(env=env)
            self.assertEqual(env, {})

    def test_module_does_not_import_torch(self):
        from util import cuda_alloc_conf

        source = pathlib.Path(cuda_alloc_conf.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported_roots)

    def test_defaults_to_os_environ_when_env_omitted(self):
        from util.cuda_alloc_conf import configure_cuda_alloc_conf

        with patch.dict(os.environ, {}, clear=True):
            configure_cuda_alloc_conf(platform="linux")
            self.assertEqual(
                os.environ.get("PYTORCH_ALLOC_CONF"),
                "expandable_segments:True",
            )

    def test_merges_existing_tokens_and_respects_explicit_expandable_segments(self):
        from util.cuda_alloc_conf import configure_cuda_alloc_conf

        env = {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8"}
        configure_cuda_alloc_conf(env=env, platform="linux")
        self.assertEqual(
            env,
            {
                "PYTORCH_ALLOC_CONF": (
                    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
                )
            },
        )
        # Idempotent on repeated invocation
        configure_cuda_alloc_conf(env=env, platform="linux")
        self.assertEqual(
            env,
            {
                "PYTORCH_ALLOC_CONF": (
                    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
                )
            },
        )

        # Handles trailing comma / whitespace cleanly
        env_ws = {"PYTORCH_ALLOC_CONF": " max_split_size_mb:128 , "}
        configure_cuda_alloc_conf(env=env_ws, platform="linux")
        self.assertEqual(
            env_ws,
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True"},
        )

        # Respects explicit user expandable_segments:False without overwriting
        env_false = {
            "PYTORCH_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:False"
        }
        configure_cuda_alloc_conf(env=env_false, platform="linux")
        self.assertEqual(
            env_false,
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:False"},
        )

    def test_merges_into_already_set_env_vars_without_creating_other_var(self):
        from util.cuda_alloc_conf import configure_cuda_alloc_conf

        # User set legacy var: merge into legacy var only
        env_legacy = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
        configure_cuda_alloc_conf(env=env_legacy, platform="linux")
        self.assertEqual(
            env_legacy,
            {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True"},
        )
        self.assertNotIn("PYTORCH_ALLOC_CONF", env_legacy)

        # User set legacy var with explicit False: preserve it
        env_legacy_false = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
        configure_cuda_alloc_conf(env=env_legacy_false, platform="linux")
        self.assertEqual(
            env_legacy_false,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
        )
        self.assertNotIn("PYTORCH_ALLOC_CONF", env_legacy_false)

        # User set both vars: merge into both without overwriting explicit values
        env_both = {
            "PYTORCH_ALLOC_CONF": "max_split_size_mb:128",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        }
        configure_cuda_alloc_conf(env=env_both, platform="linux")
        self.assertEqual(
            env_both,
            {
                "PYTORCH_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            },
        )

        # Empty string treated as unset
        env_empty = {"PYTORCH_CUDA_ALLOC_CONF": ""}
        configure_cuda_alloc_conf(env=env_empty, platform="linux")
        self.assertEqual(
            env_empty,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
        )

    def test_pytorch_import_and_cpu_execution_after_configure_has_no_warnings_or_crash(self):
        repo_root = str(pathlib.Path(__file__).resolve().parents[1])
        base_env = os.environ.copy()
        base_env.pop("PYTORCH_ALLOC_CONF", None)
        base_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        base_env["PYTHONPATH"] = repo_root

        script = "\n".join(
            [
                "import warnings",
                "from util.cuda_alloc_conf import configure_cuda_alloc_conf",
                "configure_cuda_alloc_conf()",
                "with warnings.catch_warnings(record=True) as caught:",
                "    warnings.simplefilter('always')",
                "    import torch",
                "    x = torch.ones((4, 4), device='cpu')",
                "    total = float((x @ x.T).sum().item())",
                "    assert total == 64.0, total",
                "assert not caught, f'Unexpected warnings: {[str(w.message) for w in caught]}'",
            ]
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=base_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Subprocess failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertEqual(proc.stderr.strip(), "", f"Unexpected stderr: {proc.stderr}")


class TestMainEntryPoint(unittest.TestCase):
    def test_main_calls_configure_cuda_alloc_conf_before_importing_torch(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))

        torch_import_lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
                    torch_import_lines.append(node.lineno)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "torch" or node.module.startswith("torch."):
                    torch_import_lines.append(node.lineno)

        self.assertTrue(torch_import_lines, "Expected main.py to import torch")
        earliest_torch_line = min(torch_import_lines)

        import_lines = []
        call_lines = []
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "util.cuda_alloc_conf"
                and any(alias.name == "configure_cuda_alloc_conf" for alias in stmt.names)
            ):
                import_lines.append(stmt.lineno)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Name) and func.id == "configure_cuda_alloc_conf":
                    call_lines.append(stmt.lineno)

        self.assertTrue(
            import_lines,
            "main.py must import configure_cuda_alloc_conf from util.cuda_alloc_conf",
        )
        self.assertTrue(
            call_lines,
            "main.py must call configure_cuda_alloc_conf() at module top level",
        )
        self.assertLess(
            import_lines[0],
            earliest_torch_line,
            "configure_cuda_alloc_conf import must appear before import torch",
        )
        self.assertLess(
            call_lines[0],
            earliest_torch_line,
            "configure_cuda_alloc_conf() must execute before import torch",
        )


class TestReadmeDocumentation(unittest.TestCase):
    def test_training_section_documents_expandable_segments_for_linux_slurm(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Training", readme_text)
        training_section = readme_text.split("## Training", 1)[1].split("## ", 1)[0]

        self.assertIn("PYTORCH_ALLOC_CONF=expandable_segments:True", training_section)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", training_section)
        self.assertIn("2.9", training_section)
        self.assertIn("SLURM", training_section)
        self.assertIn("Linux", training_section)


if __name__ == "__main__":
    unittest.main()
