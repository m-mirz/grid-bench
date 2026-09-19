"""A persistent `octave-cli` process, driven over its stdin and stdout.

Standard library only. Each call sends one statement, wrapped so that an
Octave error comes back as an exception here, then reads lines until an end
marker. Results travel as text lines the statement prints with `gb_out`
(prefix `GB_OUT `). Nothing timed crosses this bridge: the adapter times
inside Octave (see `SolverAdapter.clock`), so the bridge only has to be
correct, not fast.
"""
import subprocess

END = "GB_END"


class OctaveError(Exception):
    """An error raised inside Octave, with its message."""


class OctaveSession:
    def __init__(self, path_dirs: list[str]):
        self.proc = subprocess.Popen(
            ["octave-cli", "--quiet", "--norc", "--no-history", "--no-line-editing"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.eval("more('off'); page_screen_output(false);" + "".join(f"addpath('{d}');" for d in path_dirs))

    @property
    def pid(self) -> int:
        return self.proc.pid

    def eval(self, statement: str) -> list[str]:
        """Runs `statement`; returns the payload of every `GB_OUT` line."""
        self.proc.stdin.write(f"try, {statement}, catch err, printf('GB_ERROR %s\\n', "
                              f"strrep(err.message, \"\\n\", ' ')); end, printf('{END}\\n'); fflush(stdout);\n")
        self.proc.stdin.flush()
        out, error = [], None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line == END:
                break
            if line.startswith("GB_OUT "):
                out.append(line[7:])
            elif line.startswith("GB_ERROR "):
                error = line[9:]
        else:
            raise OctaveError(f"octave-cli exited (code {self.proc.wait()}) during: {statement}")
        if error is not None:
            raise OctaveError(error)
        return out

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
