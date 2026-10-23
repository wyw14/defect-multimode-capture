"""Restricted native-Structure adapter; no hidden expected values or solver code."""

import contextlib, importlib, json, sys, traceback

sys.path.insert(0, "/runner")
from codec import encode, decode

sys.path.insert(0, "/app/src")
request = decode(json.loads(sys.stdin.read()))
try:
    with contextlib.redirect_stdout(sys.stderr):
        from pymatgen.core import Structure
        import numpy as np

        initial = Structure.from_dict(request["args"][0])
        final = Structure.from_dict(request["args"][1])
        before = json.dumps(encode(request), sort_keys=True, allow_nan=False)
        native_before = json.dumps(
            [initial.as_dict(), final.as_dict()], sort_keys=True, allow_nan=False
        )
        module = importlib.import_module("vibronic_kernel")
        failure = None
        try:
            result = getattr(module, request["function"])(
                initial, final, *request["args"][2:], **request["kwargs"]
            )
        except Exception as error:
            failure = error
        unchanged = before == json.dumps(
            encode(request), sort_keys=True, allow_nan=False
        ) and native_before == json.dumps(
            [initial.as_dict(), final.as_dict()], sort_keys=True, allow_nan=False
        )
        if not unchanged:
            raise AssertionError(
                "Candidate mutated native structures or physical inputs"
            )
        if failure is not None:
            raise failure
    answer = {"ok": True, "value": encode(result)}
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    answer = {"ok": False, "error": type(error).__name__, "message": str(error)}
sys.stdout.write(json.dumps(answer, allow_nan=False))
