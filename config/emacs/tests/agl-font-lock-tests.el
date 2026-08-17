;;; agl-font-lock-tests.el --- ERT tests for agl-mode font-lock -*- lexical-binding: t; -*-

;;; Commentary:

;; Tests the structural font-lock layer of `agl-mode' (D3: no case-based
;; type coloring anywhere -- faces derive only from declaration and
;; annotation positions).  Covers: every declaration form's declared-name
;; face, soft-keyword promotion windows (both directions), contextual
;; builtins, primitive-type-annotation positions, the absence of
;; case-based coloring, identifier-boundary safety, `%{...}' interpolation
;; delimiters, and that keywords inside templates/raw-tail payloads stay
;; string-faced.  See docs/agl/reference/lexical-structure.md for the
;; promotion-window table this approximates.

;;; Code:

(require 'ert)
(require 'agl-mode)

(defmacro agl-flt--with-buffer (text &rest body)
  "Evaluate BODY in a temporary `agl-mode' buffer fontified from TEXT."
  (declare (indent 1))
  `(with-temp-buffer
     (agl-mode)
     (insert ,text)
     (font-lock-ensure)
     (goto-char (point-min))
     ,@body))

(defun agl-flt--pos-after (needle)
  "Return the buffer position right after the first occurrence of NEEDLE."
  (save-excursion
    (goto-char (point-min))
    (search-forward needle)
    (point)))

(defun agl-flt--pos-before (needle)
  "Return the buffer position of the first occurrence of NEEDLE."
  (save-excursion
    (goto-char (point-min))
    (search-forward needle)
    (- (point) (length needle))))

(defun agl-flt--face-at (pos)
  "Return the `face' text property at POS."
  (get-text-property pos 'face))

(defun agl-flt--face-of (needle)
  "Return the `face' text property of the first character of NEEDLE."
  (agl-flt--face-at (agl-flt--pos-before needle)))

;; --- Declaration forms: the declared name gets the right face ---

(ert-deftest agl-flt-def-name-is-function-face ()
  (agl-flt--with-buffer "def greet(x: text) -> text = x\n"
    (should (eq (agl-flt--face-of "greet") 'font-lock-function-name-face))))

(ert-deftest agl-flt-program-def-name-is-function-face ()
  (agl-flt--with-buffer "program def main() -> unit =\n  print \"hi\"\n"
    (should (eq (agl-flt--face-of "main") 'font-lock-function-name-face))))

(ert-deftest agl-flt-extern-def-name-is-function-face ()
  (agl-flt--with-buffer "extern def to_slug(title: text) -> text\n"
    (should (eq (agl-flt--face-of "to_slug") 'font-lock-function-name-face))))

(ert-deftest agl-flt-builtin-def-name-is-function-face ()
  (agl-flt--with-buffer "builtin def copy[T](value: T) -> T\n"
    (should (eq (agl-flt--face-of "copy") 'font-lock-function-name-face))))

(ert-deftest agl-flt-qualified-def-name-faces-only-terminal-segment ()
  (agl-flt--with-buffer "def Box::get[E](self) -> E = self.value\n"
    (should (eq (agl-flt--face-of "get") 'font-lock-function-name-face))
    ;; The qualifier segment "Box" is not a declaration position (D3):
    ;; it stays unfaced.
    (should-not (eq (agl-flt--face-of "Box") 'font-lock-function-name-face))
    (should-not (eq (agl-flt--face-of "Box") 'font-lock-type-face))))

(ert-deftest agl-flt-record-name-is-type-face ()
  (agl-flt--with-buffer "record Point(x: int, y: int)\n"
    (should (eq (agl-flt--face-of "Point") 'font-lock-type-face))))

(ert-deftest agl-flt-enum-name-is-type-face ()
  (agl-flt--with-buffer "enum Review\n  | Pass\n  | Fail\n"
    (should (eq (agl-flt--face-of "Review") 'font-lock-type-face))))

(ert-deftest agl-flt-type-alias-name-is-type-face ()
  (agl-flt--with-buffer "type Status = Review\n"
    (should (eq (agl-flt--face-of "Status") 'font-lock-type-face))))

(ert-deftest agl-flt-exception-name-is-type-face ()
  (agl-flt--with-buffer "exception Retryable extends Exception\n"
    (should (eq (agl-flt--face-of "Retryable") 'font-lock-type-face))))

(ert-deftest agl-flt-let-name-is-variable-face ()
  (agl-flt--with-buffer "let count = 1\n"
    (should (eq (agl-flt--face-of "count") 'font-lock-variable-name-face))))

(ert-deftest agl-flt-var-name-is-variable-face ()
  (agl-flt--with-buffer "var total = 0\n"
    (should (eq (agl-flt--face-of "total") 'font-lock-variable-name-face))))

(ert-deftest agl-flt-param-name-is-variable-face ()
  (agl-flt--with-buffer "param retries: int = 3\n"
    (should (eq (agl-flt--face-of "retries") 'font-lock-variable-name-face))))

(ert-deftest agl-flt-catch-as-alias-is-variable-face ()
  (agl-flt--with-buffer
      (concat "try\n"
              "  risky()\n"
              "catch NotFound as e =>\n"
              "  print e\n")
    (should (eq (agl-flt--face-of "e =>") 'font-lock-variable-name-face))
    ;; The exception-type pattern name is not a binder (only the "as"
    ;; alias binds -- see docs/agl/reference/exceptions.md) and stays
    ;; unfaced.
    (should-not (eq (agl-flt--face-of "NotFound") 'font-lock-variable-name-face))))

(ert-deftest agl-flt-catch-without-as-has-no-binder-face ()
  (agl-flt--with-buffer
      (concat "try\n"
              "  risky()\n"
              "catch NotFound =>\n"
              "  print 1\n")
    (should-not (eq (agl-flt--face-of "NotFound") 'font-lock-variable-name-face))))

;; --- Soft keywords: promotion windows, both directions ---

(ert-deftest agl-flt-import-at-item-start-is-keyword-face ()
  (agl-flt--with-buffer "import foo/bar\n"
    (should (eq (agl-flt--face-of "import") 'font-lock-keyword-face))))

(ert-deftest agl-flt-import-not-at-item-start-is-unfaced ()
  (agl-flt--with-buffer "let import = 1\n"
    (should-not (eq (agl-flt--face-of "import") 'font-lock-keyword-face))))

(ert-deftest agl-flt-open-at-item-start-is-keyword-face ()
  (agl-flt--with-buffer "open Geometry\n"
    (should (eq (agl-flt--face-of "open") 'font-lock-keyword-face))))

(ert-deftest agl-flt-open-not-at-item-start-is-unfaced ()
  (agl-flt--with-buffer "let open = 1\n"
    (should-not (eq (agl-flt--face-of "open") 'font-lock-keyword-face))))

(ert-deftest agl-flt-import-directly-after-open-is-keyword-face ()
  (agl-flt--with-buffer "open import foo/bar\n"
    (should (eq (agl-flt--face-of "open") 'font-lock-keyword-face))
    (should (eq (agl-flt--face-of "import") 'font-lock-keyword-face))))

(ert-deftest agl-flt-export-at-item-start-is-keyword-face ()
  (agl-flt--with-buffer "export foo\n"
    (should (eq (agl-flt--face-of "export") 'font-lock-keyword-face))))

(ert-deftest agl-flt-export-not-at-item-start-is-unfaced ()
  (agl-flt--with-buffer "let export = \"hello\"\n"
    (should-not (eq (agl-flt--face-of "export") 'font-lock-keyword-face))))

(ert-deftest agl-flt-using-within-import-line-is-keyword-face ()
  (agl-flt--with-buffer "import foo/bar using thing\n"
    (should (eq (agl-flt--face-of "using") 'font-lock-keyword-face))))

(ert-deftest agl-flt-using-not-in-declaration-line-is-unfaced ()
  (agl-flt--with-buffer "let using = \"hello\"\n"
    (should-not (eq (agl-flt--face-of "using") 'font-lock-keyword-face))))

(ert-deftest agl-flt-hiding-within-open-line-is-keyword-face ()
  (agl-flt--with-buffer "open Geometry hiding origin\n"
    (should (eq (agl-flt--face-of "hiding") 'font-lock-keyword-face))))

(ert-deftest agl-flt-hiding-not-in-declaration-line-is-unfaced ()
  (agl-flt--with-buffer "let hiding = 1\n"
    (should-not (eq (agl-flt--face-of "hiding") 'font-lock-keyword-face))))

(ert-deftest agl-flt-scope-at-item-start-is-keyword-face ()
  (agl-flt--with-buffer "scope Geometry\n  def area() -> int = 0\nend Geometry\n"
    (should (eq (agl-flt--face-of "scope") 'font-lock-keyword-face))))

(ert-deftest agl-flt-scope-not-at-item-start-is-unfaced ()
  (agl-flt--with-buffer "let scope = 1\n"
    (should-not (eq (agl-flt--face-of "scope") 'font-lock-keyword-face))))

(ert-deftest agl-flt-end-with-closer-path-is-keyword-face ()
  (agl-flt--with-buffer "scope Geometry\n  def area() -> int = 0\nend Geometry\n"
    (should (eq (agl-flt--face-of "end Geometry") 'font-lock-keyword-face))))

(ert-deftest agl-flt-end-as-field-name-is-unfaced ()
  (agl-flt--with-buffer "record R(end: int)\n"
    (should-not (eq (agl-flt--face-of "end") 'font-lock-keyword-face))))

(ert-deftest agl-flt-end-without-closer-path-is-unfaced ()
  (agl-flt--with-buffer "let x = 1\nend\n"
    (should-not (eq (agl-flt--face-of "end\n") 'font-lock-keyword-face))))

;; --- Contextual builtins and raw-tail names ---

(ert-deftest agl-flt-print-is-builtin-face ()
  (agl-flt--with-buffer "print \"hi\"\n"
    (should (eq (agl-flt--face-of "print") 'font-lock-builtin-face))))

(ert-deftest agl-flt-ask-is-builtin-face ()
  (agl-flt--with-buffer "let r = ask(\"hi\")\n"
    (should (eq (agl-flt--face-of "ask") 'font-lock-builtin-face))))

(ert-deftest agl-flt-exec-is-builtin-face ()
  (agl-flt--with-buffer "let r = exec(\"ls\")\n"
    (should (eq (agl-flt--face-of "exec") 'font-lock-builtin-face))))

(ert-deftest agl-flt-exec-bang-is-builtin-face ()
  (agl-flt--with-buffer "exec! ls -la\n"
    (should (eq (agl-flt--face-of "exec!") 'font-lock-builtin-face))))

(ert-deftest agl-flt-ask-bang-is-builtin-face ()
  (agl-flt--with-buffer "ask! Summarize this\n"
    (should (eq (agl-flt--face-of "ask!") 'font-lock-builtin-face))))

;; --- Primitive type-annotation positions (contextual, D3) ---

(ert-deftest agl-flt-primitive-type-after-colon-is-type-face ()
  (agl-flt--with-buffer "def f(x: text) -> text = x\n"
    (should (eq (agl-flt--face-of "text) ->") 'font-lock-type-face))))

(ert-deftest agl-flt-primitive-type-after-arrow-is-type-face ()
  (agl-flt--with-buffer "def f(x: text) -> int = 1\n"
    (should (eq (agl-flt--face-of "int = 1") 'font-lock-type-face))))

(ert-deftest agl-flt-primitive-name-elsewhere-is-not-type-face ()
  (agl-flt--with-buffer "let text = 1\n"
    (should-not (eq (agl-flt--face-of "text") 'font-lock-type-face))
    ;; It is an ordinary `let'-bound variable name instead.
    (should (eq (agl-flt--face-of "text") 'font-lock-variable-name-face))))

;; --- D3: no case-based coloring ---

(ert-deftest agl-flt-no-case-based-coloring-bare-operand-unfaced ()
  (agl-flt--with-buffer "let x = Option\n"
    (should-not (eq (agl-flt--face-of "Option") 'font-lock-type-face))))

(ert-deftest agl-flt-no-case-based-coloring-lowercase-record-is-type-face ()
  (agl-flt--with-buffer "record option(value: int)\n"
    (should (eq (agl-flt--face-of "option") 'font-lock-type-face))))

;; --- Keyword/identifier boundary safety ---

(ert-deftest agl-flt-keyword-not-matched-inside-hyphenated-identifier ()
  (agl-flt--with-buffer "let ask-prompt = 1\n"
    (should-not (eq (agl-flt--face-of "ask-prompt") 'font-lock-builtin-face))
    (should (eq (agl-flt--face-of "ask-prompt") 'font-lock-variable-name-face))))

(ert-deftest agl-flt-keyword-not-matched-inside-plus-joined-identifier ()
  (agl-flt--with-buffer "let a+and+b = 1\n"
    (should-not (eq (agl-flt--face-of "and+b") 'font-lock-keyword-face))))

;; --- `%{...}' interpolation delimiters ---

(ert-deftest agl-flt-interpolation-delimiters-faced-in-template ()
  (agl-flt--with-buffer "let a = \"cost is %{price}\"\n"
    (should (eq (agl-flt--face-of "%{price}") 'agl-interpolation-face))
    (should (eq (agl-flt--face-at (1- (agl-flt--pos-after "%{price}"))) 'agl-interpolation-face))))

(ert-deftest agl-flt-interpolation-hole-contents-stay-string-faced ()
  (agl-flt--with-buffer "let a = \"cost is %{price}\"\n"
    (should (eq (agl-flt--face-of "price") 'font-lock-string-face))))

(ert-deftest agl-flt-interpolation-delimiters-faced-in-raw-tail-payload ()
  (agl-flt--with-buffer "ask! Summarize %{topic} please\n"
    (should (eq (agl-flt--face-of "%{topic}") 'agl-interpolation-face))))

(ert-deftest agl-flt-escaped-percent-brace-is-not-faced ()
  (agl-flt--with-buffer "let a = \"cost is \\%{100}\"\n"
    (should-not (eq (agl-flt--face-of "%{100}") 'agl-interpolation-face))
    (should (eq (agl-flt--face-of "%{100}") 'font-lock-string-face))))

;; --- Keywords inside templates/raw-tail payloads stay string-faced ---

(ert-deftest agl-flt-keyword-inside-template-stays-string-faced ()
  (agl-flt--with-buffer "let a = \"let x = 1\"\n"
    (should (eq (agl-flt--face-of "let x") 'font-lock-string-face))))

(ert-deftest agl-flt-keyword-inside-raw-tail-payload-stays-string-faced ()
  (agl-flt--with-buffer "exec! let x = 1\n"
    (should (eq (agl-flt--face-of "let x") 'font-lock-string-face))))

;; --- Numbers ---

(ert-deftest agl-flt-int-literal-is-number-face ()
  (agl-flt--with-buffer "let x = 42\n"
    (should (eq (agl-flt--face-of "42") agl--number-face))))

(ert-deftest agl-flt-decimal-literal-is-number-face ()
  (agl-flt--with-buffer "let x = 3.14\n"
    (should (eq (agl-flt--face-of "3.14") agl--number-face))))

(ert-deftest agl-flt-digits-inside-identifier-are-not-number-faced ()
  ;; `a1b' is one identifier: digits within it are not a numeric literal.
  (agl-flt--with-buffer "let a1b = 1\n"
    (should-not (eq (agl-flt--face-at (agl-flt--pos-before "1b")) agl--number-face))))

;; --- Operators, including `::' (Fix 5) ---

(ert-deftest agl-flt-single-char-operator-is-operator-face ()
  (agl-flt--with-buffer "let z = 1 + 2\n"
    (should (eq (agl-flt--face-of "+") agl--operator-face))))

(ert-deftest agl-flt-multi-char-operator-is-operator-face ()
  (agl-flt--with-buffer "let ok = a <= b\n"
    (should (eq (agl-flt--face-of "<=") agl--operator-face))))

(ert-deftest agl-flt-double-colon-operator-is-operator-face ()
  (agl-flt--with-buffer "let x = a::b\n"
    (should (eq (agl-flt--face-of "::") agl--operator-face))))

(ert-deftest agl-flt-assignment-operator-is-operator-faced ()
  (agl-flt--with-buffer "r := 1\n"
    (should (eq (agl-flt--face-of ":=") agl--operator-face))))

(ert-deftest agl-flt-unspaced-plus-is-part-of-identifier ()
  ;; `a+b' is a single identifier, so its `+' is not an operator.
  (agl-flt--with-buffer "let s = a+b\n"
    (should-not (eq (agl-flt--face-of "+") agl--operator-face))))

;; --- Zone markers `@pos'/`@std'/`@named' ---

(ert-deftest agl-flt-zone-marker-pos-is-keyword-face ()
  (agl-flt--with-buffer "def f(@pos, x: int) -> int = x\n"
    (should (eq (agl-flt--face-of "@pos") 'font-lock-keyword-face))))

(ert-deftest agl-flt-zone-marker-std-is-keyword-face ()
  (agl-flt--with-buffer "def f(x: int, @std, y: int) -> int = x + y\n"
    (should (eq (agl-flt--face-of "@std") 'font-lock-keyword-face))))

(ert-deftest agl-flt-zone-marker-named-is-keyword-face ()
  (agl-flt--with-buffer "def f(x: int, @named, y: int) -> int = x + y\n"
    (should (eq (agl-flt--face-of "@named") 'font-lock-keyword-face))))

(ert-deftest agl-flt-unknown-at-name-is-not-a-zone-marker ()
  (agl-flt--with-buffer "def f(x: int, @nope, y: int) -> int = x\n"
    (should-not (eq (agl-flt--face-of "@nope") 'font-lock-builtin-face))))

;; --- A keyword immediately after a delimiter or operator is still faced ---

(ert-deftest agl-flt-keyword-immediately-after-open-paren-is-faced ()
  (agl-flt--with-buffer "(let x = 1)\n"
    (should (eq (agl-flt--face-of "let") 'font-lock-keyword-face))))

(ert-deftest agl-flt-keyword-immediately-after-open-bracket-is-faced ()
  (agl-flt--with-buffer "[if x]\n"
    (should (eq (agl-flt--face-of "if") 'font-lock-keyword-face))))

(ert-deftest agl-flt-keyword-immediately-after-comma-is-faced ()
  (agl-flt--with-buffer "let ok = f(x,and,y)\n"
    (should (eq (agl-flt--face-of "and") 'font-lock-keyword-face))))

;; --- `as?' is a single lexeme ---

(ert-deftest agl-flt-as-optional-is-single-lexeme-keyword-face ()
  (agl-flt--with-buffer "let ok = v as? int\n"
    (should (eq (agl-flt--face-of "as?") 'font-lock-keyword-face))
    ;; The trailing `?' is part of the same match, not left unfaced.
    (should (eq (agl-flt--face-at (1- (agl-flt--pos-after "as?"))) 'font-lock-keyword-face))))

;; --- Backslash runs before `%{': template parity vs. raw-tail semantics
;;     (Fix 4) ---

(ert-deftest agl-flt-double-backslash-before-interpolation-is-not-escaped-in-template ()
  (agl-flt--with-buffer "let a = \"cost is \\\\%{100}\"\n"
    ;; An even run pairs off: the hole is unescaped.
    (should (eq (agl-flt--face-of "%{100}") 'agl-interpolation-face))))

(ert-deftest agl-flt-triple-backslash-before-interpolation-is-escaped-in-template ()
  (agl-flt--with-buffer "let a = \"cost is \\\\\\%{100}\"\n"
    ;; An odd run leaves one unpaired backslash: the hole is escaped.
    (should-not (eq (agl-flt--face-of "%{100}") 'agl-interpolation-face))
    (should (eq (agl-flt--face-of "%{100}") 'font-lock-string-face))))

(ert-deftest agl-flt-single-backslash-before-interpolation-is-escaped-in-raw-tail ()
  (agl-flt--with-buffer "exec! echo \\%{100}\n"
    (should-not (eq (agl-flt--face-of "%{100}") 'agl-interpolation-face))))

(ert-deftest agl-flt-double-backslash-before-interpolation-is-still-escaped-in-raw-tail ()
  (agl-flt--with-buffer "exec! echo \\\\%{100}\n"
    ;; Unlike a template, a raw-tail payload owns its backslashes: any
    ;; single immediately preceding `\' escapes the hole, with no parity
    ;; counting, so a run of two is still escaped here.
    (should-not (eq (agl-flt--face-of "%{100}") 'agl-interpolation-face))))

;; --- The interpolation face cannot leak outside its string region (Fix 3) ---

(ert-deftest agl-flt-interpolation-face-does-not-leak-past-string-region ()
  (agl-flt--with-buffer "exec! echo %{p\nlet c = f(1)}\n"
    ;; The `%{' on the exec! line is unbalanced within that line's inline
    ;; raw-tail payload; the `}' on the next, ordinary code line must not
    ;; be painted with `agl-interpolation-face'.
    (should-not (eq (agl-flt--face-of "}") 'agl-interpolation-face))))

(ert-deftest agl-flt-close-brace-without-hole-is-not-faced ()
  (agl-flt--with-buffer "let a = { x = 1 }\n"
    (should-not (eq (agl-flt--face-of "}") 'agl-interpolation-face))))

;; --- `end' promotion works at any indentation, not just column 0 ---

(ert-deftest agl-flt-end-promoted-at-indented-item-start ()
  (agl-flt--with-buffer
      (concat "scope Outer\n"
              "  scope Inner\n"
              "    def f() -> int = 0\n"
              "  end Inner\n"
              "end Outer\n")
    (should (eq (agl-flt--face-of "end Inner") 'font-lock-keyword-face))))

;; --- Regression: user-defined types in annotation position (Fix 1) ---

(ert-deftest agl-flt-user-defined-types-faced-in-annotation-position ()
  (agl-flt--with-buffer "def f(p: Point) -> Review = p\n"
    (should (eq (agl-flt--face-of "Point") 'font-lock-type-face))
    (should (eq (agl-flt--face-of "Review") 'font-lock-type-face))))

(ert-deftest agl-flt-module-route-qualified-type-faces-terminal-segment-only ()
  (agl-flt--with-buffer "def f(p: foo/bar::Point) -> int = 1\n"
    (should (eq (agl-flt--face-of "Point") 'font-lock-type-face))
    ;; The module-route segments are not a declaration position (D3):
    ;; they stay unfaced, matching how a qualifier prefix is treated
    ;; everywhere else in this file.
    (should-not (eq (agl-flt--face-of "foo") 'font-lock-type-face))
    (should-not (eq (agl-flt--face-of "bar") 'font-lock-type-face))))

(ert-deftest agl-flt-generic-type-head-in-annotation-is-type-faced ()
  (agl-flt--with-buffer "def f(o: Option[int]) -> int = 1\n"
    (should (eq (agl-flt--face-of "Option") 'font-lock-type-face))))

(ert-deftest agl-flt-user-type-in-field-annotation-is-type-faced ()
  (agl-flt--with-buffer "record R(a: Point, b: text)\n"
    (should (eq (agl-flt--face-of "Point") 'font-lock-type-face))
    (should (eq (agl-flt--face-of "text") 'font-lock-type-face))))

;; --- Regression: a constructor pattern leaves the constructor unfaced
;;     (Fix 7) ---

(ert-deftest agl-flt-let-constructor-pattern-does-not-face-constructor ()
  (agl-flt--with-buffer "let Point(x, y) = p\n"
    (should-not (eq (agl-flt--face-of "Point") 'font-lock-variable-name-face))))

(ert-deftest agl-flt-let-qualified-nullary-pattern-is-not-variable-faced ()
  (agl-flt--with-buffer "let A::x() = e\n"
    (should-not (eq (agl-flt--face-of "x()") 'font-lock-variable-name-face))))

;; --- Annotation position is contextual, not a bare `:' anywhere ---

(ert-deftest agl-flt-dict-literal-value-is-not-type-faced ()
  ;; `{ key: value }' is a dict literal: the `:' separates an entry, so the
  ;; value is an ordinary expression rather than a type annotation.
  (agl-flt--with-buffer "let d = { foo: bar }\n"
    (should-not (eq (agl-flt--face-of "bar") 'font-lock-type-face))))

(ert-deftest agl-flt-primitive-name-as-dict-value-is-not-type-faced ()
  (agl-flt--with-buffer "let d = { foo: text }\n"
    (should-not (eq (agl-flt--face-of "text") 'font-lock-type-face))))

(ert-deftest agl-flt-primitive-name-as-binder-is-not-type-faced ()
  (agl-flt--with-buffer "let text = 1\n"
    (should-not (eq (agl-flt--face-of "text") 'font-lock-type-face))))

(ert-deftest agl-flt-plain-let-binder-is-still-variable-faced ()
  (agl-flt--with-buffer "let plain = 1\n"
    (should (eq (agl-flt--face-of "plain") 'font-lock-variable-name-face))))

(provide 'agl-font-lock-tests)
;;; agl-font-lock-tests.el ends here
