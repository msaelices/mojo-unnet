# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import sys
import tempfile
from pathlib import Path

try:
    # Don't require including IPython as a dependency
    from IPython.core.magic import register_cell_magic  # type: ignore
    from IPython.display import display
except ImportError:
    display = None

    def register_cell_magic(fn):  # noqa: ANN001, ANN201
        return fn


from .paths import MojoCompilationError
from .run import subprocess_run_mojo


@register_cell_magic
def mojo(line, cell) -> None:  # noqa: ANN001
    """Execute Mojo code in a Jupyter notebook cell.

    Behaves like a normal Python cell - executes the Mojo code and captures
    output. Mojo's main() function must return None, so use print() for output
    or call Python functions directly.

    Usage:
        ```mojo
        %%mojo
        fn main():
            print("Hello from Mojo!")
        ```

        ```mojo
        %%mojo
        from python import Python

        fn main():
            let json = Python.import_module("json")
            let data = Dict[String, String]()
            data["key"] = "value"
            print(json.dumps(data))
        ```
    """
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        mojo_path = path / "cell.mojo"

        # Write the Mojo code to a temporary file
        with open(mojo_path, "w") as f:
            f.write(cell)

        # Create an __init__.mojo file for proper module structure
        (path / "__init__.mojo").touch()

        # Execute the Mojo code
        command = ["run", str(mojo_path)]
        result = subprocess_run_mojo(command, capture_output=True)

        if not result.returncode:
            # Display stdout from Mojo execution
            stdout = result.stdout.decode()
            if stdout.strip():
                print(stdout, end="")

            # For PythonObject display, we rely on the Mojo code to use
            # Python interop appropriately. The execution captures stdout
            # which is the primary output mechanism for now.
            # Future enhancement could capture return values from main().

        else:
            raise MojoCompilationError(
                mojo_path,
                command,
                result.stdout.decode(),
                result.stderr.decode(),
            )
