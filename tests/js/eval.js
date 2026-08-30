/* Runs one expression against a frontend script and prints the result as JSON.

   The bridge tests/test_frontend_js.py uses to execute the frontend's own code
   instead of grepping its source. Both scripts it loads (viewer/lib.js and
   review_ui/affine.js) are plain classic scripts that export to node when
   there is a `module` to export to.

   A promise is awaited, so an expression may be asynchronous; anything else
   passes through unchanged. A rejection exits non-zero with its stack rather
   than printing `{}`, which would read as an empty result.

   argv: <script path> <expression>, with the script's exports bound to `L`. */
const path = require("path");
const L = require(path.resolve(process.argv[2]));
const value = new Function("L", "return (" + process.argv[3] + ");")(L);
Promise.resolve(value).then(
  (resolved) => process.stdout.write(JSON.stringify(resolved === undefined ? null : resolved)),
  (err) => {
    process.stderr.write(String((err && err.stack) || err));
    process.exit(1);
  },
);
