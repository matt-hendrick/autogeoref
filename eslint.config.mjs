// The frontend's linter. Dev-only: see package.json for why an npm dependency
// is allowed here and not in the deploy path.
//
// Every surface is a plain classic script — no modules, no bundler — so the
// scripts share globals with each other and with the vendored libraries. That
// is declared per directory below rather than guessed, which is what makes
// `no-undef` usable: since the viewer became four files sharing one global
// namespace, a name lib.js stops exporting, or a script tag in the wrong
// order, is otherwise a blank page no test in this repo can see.
import css from "@eslint/css";
import js from "@eslint/js";
import globals from "globals";
import html from "eslint-plugin-html";
import noUnsanitized from "eslint-plugin-no-unsanitized";

const RULES = {
  ...js.configs.recommended.rules,
  ...noUnsanitized.configs.recommended.rules,
  eqeqeq: ["error", "smart"],
  "no-var": "error",
  "prefer-const": "error",
  // These pages are 600-950 line IIFEs whose helpers reuse short names (`m`,
  // `s`, `b`); a shadowed one reads as the outer value and is not.
  "no-shadow": "error",
  // Two exemptions, both deliberate. `catch (e) { /* … */ }` names the failure
  // being swallowed and the comment is the point. `_b` is the ignored argument
  // of a handler in a dispatch table whose siblings need it — a uniform
  // signature, not a leftover.
  "no-unused-vars": ["error", { caughtErrors: "none", argsIgnorePattern: "^_" }],
};

const script = (extraGlobals) => ({
  // `html` extracts the inline <script> of a page that has one; no-unsanitized
  // is what makes a markup write cost a written reason on every surface.
  plugins: { html, "no-unsanitized": noUnsanitized },
  languageOptions: {
    ecmaVersion: 2022,
    sourceType: "script",
    globals: {
      ...globals.browser,
      // the export tail both shared scripts carry, so a test can run them
      module: "readonly",
      ...extraGlobals,
    },
  },
  rules: RULES,
});

export default [
  {
    // the vendored map libraries and every tree that is data, a build output,
    // or somebody else's package — this lints what this repo hand-wrote
    ignores: [
      "viewer/vendor/**",
      "node_modules/**",
      ".venv/**",
      "venv/**",
      ".claude/**",
      "cache/**",
      "deploy/**",
      "work/**",
      "fixtures/**",
    ],
  },
  {
    // the .html glob is not redundant: it is what stops a future inline
    // <script> from re-entering the page unlinted
    files: ["viewer/*.js", "viewer/*.html"],
    ...script({
      maplibregl: "readonly",
      pmtiles: "readonly",
      ViewerLib: "readonly",
      // config.js sets this; the deploy bundle rewrites it with a real token
      MAPBOX_TOKEN: "readonly",
    }),
  },
  {
    files: ["src/autogeoref/*_ui/*.js", "src/autogeoref/*_ui/*.html"],
    ...script({ maplibregl: "readonly", ReviewAffine: "readonly", Board: "readonly" }),
  },
  {
    // the files that DEFINE those globals: declaring them here as well would
    // make each one a redeclaration of itself
    files: ["viewer/lib.js", "src/autogeoref/*_ui/affine.js", "src/autogeoref/*_ui/board.js"],
    languageOptions: { globals: { ViewerLib: "off", ReviewAffine: "off", Board: "off" } },
  },
  {
    files: ["tests/js/*.js"],
    plugins: { "no-unsanitized": noUnsanitized },
    languageOptions: { ecmaVersion: 2022, sourceType: "script", globals: globals.node },
    rules: RULES,
  },
  {
    files: ["viewer/*.css", "src/autogeoref/*_ui/*.css"],
    language: "css/css",
    ...css.configs.recommended,
    rules: {
      ...css.configs.recommended.rules,
      // A browser-support policy, not a correctness check. Every property it
      // objects to here (backdrop-filter, scrollbar-width, ui-monospace) is
      // decoration that degrades to a plain panel, and the atlas is the
      // product. Syntax and duplicate-selector checking stays on.
      "css/use-baseline": "off",
    },
  },
];
