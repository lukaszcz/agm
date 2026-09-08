;;; agl-indent-tests.el --- ERT tests for agl-mode indentation -*- lexical-binding: t; -*-

;;; Commentary:

;; Scenario tables for the AgL indentation engine: block opening and
;; closing, branch-marker alignment for every marker, scope regions and
;; their closers, bracket continuation, TAB cycling, electric re-indent,
;; and the immunity of raw-tail payloads and multi-line templates.  See the layout rules in
;; docs/agl/reference/lexical-structure.md.

;;; Code:

(require 'ert)
(require 'agl-mode)
(require 'agl-indent)

(defun agl-ind--indent-of (text line)
  "Return the column `agl-indent-line' computes for LINE of TEXT.

TEXT is inserted into an `agl-mode' buffer and LINE (1-based) is
indented; the resulting indentation column is returned."
  (with-temp-buffer
    (agl-mode)
    (insert text)
    (goto-char (point-min))
    (forward-line (1- line))
    (agl-indent-line)
    (current-indentation)))

(defun agl-ind--reindented (text)
  "Return TEXT after `indent-region' over an `agl-mode' buffer."
  (with-temp-buffer
    (agl-mode)
    (insert text)
    (indent-region (point-min) (point-max))
    (buffer-substring-no-properties (point-min) (point-max))))

(defun agl-ind--typed (keys)
  "Return the text of an `agl-mode' buffer after KEYS are typed into it.

Every character is inserted as if typed, so `post-self-insert-hook' runs
as it does interactively and `electric-indent-mode' sees each key.  A
`\\r' in KEYS stands for the user erasing the indentation the mode just
put on the line, which is how a declaration gets written at its own
level under a deeper body."
  (with-temp-buffer
    (agl-mode)
    (electric-indent-local-mode 1)
    (dolist (key (string-to-list keys))
      (let ((last-command-event key))
        (cond ((eq key ?\r) (delete-region (line-beginning-position) (point)))
              ((eq key ?\n) (call-interactively #'newline))
              (t (call-interactively #'self-insert-command)))))
    (buffer-substring-no-properties (point-min) (point-max))))

;; --- Block opening ---

(ert-deftest agl-ind-body-after-def-is-indented ()
  (should (= (agl-ind--indent-of "def f() -> int =\nx\n" 2) 2)))

(ert-deftest agl-ind-body-after-record-header-is-indented ()
  (should (= (agl-ind--indent-of "record R\nx: int\n" 2) 2)))

(ert-deftest agl-ind-var-field-indents-like-a-plain-field ()
  ;; A `var' field marker declares a field, not a nested block, so the line
  ;; after it stays at the field level.
  (should (= (agl-ind--indent-of "record R\nvar x: int\n" 2) 2))
  (should (= (agl-ind--indent-of "record R\n  var x: int\ny: int\n" 3) 2)))

(ert-deftest agl-ind-body-after-arrow-is-indented ()
  (should (= (agl-ind--indent-of "let v = if\n  | a =>\nb\n" 3) 4)))

(ert-deftest agl-ind-body-after-of-is-indented ()
  (should (= (agl-ind--indent-of "case v of\n| A => 1\n" 2) 2)))

(ert-deftest agl-ind-inline-body-does-not-open-a-block ()
  ;; `def f() -> int = 1' has its body inline, so the next line is a
  ;; sibling declaration rather than a nested block.
  (should (= (agl-ind--indent-of "def f() -> int = 1\ndef g() -> int = 2\n" 2) 0)))

(ert-deftest agl-ind-name-ending-in-a-keyword-does-not-open-a-block ()
  ;; `registry' ends in the letters of `try', `undo' in `do', and `motif'
  ;; in `if'; none of them is that keyword, so the next line is a sibling.
  (should (= (agl-ind--indent-of "let r = registry\nlet s = 1\n" 2) 0))
  (should (= (agl-ind--indent-of "let r = undo\nlet s = 1\n" 2) 0))
  (should (= (agl-ind--indent-of "let g = motif\nlet s = 1\n" 2) 0)))

(ert-deftest agl-ind-keyword-after-an-operator-still-opens-a-block ()
  ;; An operator run never begins an identifier, so `=>do' lexes as the
  ;; arrow and the keyword; `a->do', by contrast, is one whole name.
  (should (= (agl-ind--indent-of "let f = if | a =>do\n  step()\n" 2) 2))
  (should (= (agl-ind--indent-of "let f = a->do\nlet s = 1\n" 2) 0)))

(ert-deftest agl-ind-header-keyword-inside-a-line-does-not-open-a-block ()
  ;; A header keyword is only a header as the line's first token: here `do'
  ;; is string content passed to `print'.
  (should (= (agl-ind--indent-of "print(\"do it\")\nprint(\"next\")\n" 2) 0))
  (should (= (agl-ind--indent-of "print(\"case it\")\nprint(\"next\")\n" 2) 0)))

(ert-deftest agl-ind-trailing-comparison-does-not-open-a-block ()
  ;; `>=' is one operator token, so its `=' is not the assignment that
  ;; introduces a suite.
  (should (= (agl-ind--indent-of "let ok = a >=\nlet s = 1\n" 2) 0))
  (should (= (agl-ind--indent-of "let ok = a !=\nlet s = 1\n" 2) 0)))

(ert-deftest agl-ind-trailing-assignment-opens-a-block ()
  ;; `:=' is destructive assignment, and `=' preceded by a name is the
  ;; ordinary one; both introduce a suite.
  (should (= (agl-ind--indent-of "count :=\n  1\n" 2) 2))
  (should (= (agl-ind--indent-of "let v =\n  1\n" 2) 2)))

(ert-deftest agl-ind-name-ending-in-an-arrow-does-not-open-a-block ()
  ;; `a->' is a single AgL identifier: `-' and `>' both continue a name.
  (should (= (agl-ind--indent-of "let f = a->\nlet s = 1\n" 2) 0)))

(ert-deftest agl-ind-name-ending-in-a-raw-tail-keyword-does-not-open-a-block ()
  ;; `do-exec$' is one identifier, not the `exec$' raw-tail opener.
  (should (= (agl-ind--indent-of "let x = do-exec$\nlet s = 1\n" 2) 0)))

;; --- Continuation of the previous line's level ---

(ert-deftest agl-ind-sibling-statement-keeps-indentation ()
  (should (= (agl-ind--indent-of "def f() -> unit =\n  print \"a\"\nprint \"b\"\n" 3) 2)))

(ert-deftest agl-ind-first-line-is-column-zero ()
  (should (= (agl-ind--indent-of "def f() -> int = 1\n" 1) 0)))

;; --- Branch-marker alignment, every marker ---

(ert-deftest agl-ind-pipe-marker-indents-under-its-header ()
  ;; A `|' branch belongs to the construct's body, as stdlib writes enum
  ;; variants and `if' branches.
  (should (= (agl-ind--indent-of "let v = if\n| a => 1\n" 2) 2)))

(ert-deftest agl-ind-else-marker-aligns-with-owner ()
  (should (= (agl-ind--indent-of "if\n  | a => 1\nelse => 2\n" 3) 0)))

(ert-deftest agl-ind-until-marker-aligns-with-owner ()
  (should (= (agl-ind--indent-of "do\n  step()\nuntil done?\n" 3) 0)))

(ert-deftest agl-ind-done-marker-aligns-with-owner ()
  (should (= (agl-ind--indent-of "while cond\n  step()\ndone\n" 3) 0)))

(ert-deftest agl-ind-catch-marker-aligns-with-owner ()
  (should (= (agl-ind--indent-of "try\n  risky()\ncatch e => 1\n" 3) 0)))

(ert-deftest agl-ind-nested-marker-aligns-with-inner-owner ()
  (should (= (agl-ind--indent-of
              "def f() -> int =\n  try\n    risky()\n  catch e => 1\n" 4)
             2)))

(ert-deftest agl-ind-marker-is-not-a-longer-identifier ()
  ;; `done-with' is one AgL identifier, not the `done' marker.
  (should (= (agl-ind--indent-of "def f() -> unit =\n  done-with()\n" 2) 2)))

;; --- Symbolic continuation lines ---

(ert-deftest agl-ind-wrapped-return-type-indents-under-its-signature ()
  (should (= (agl-ind--indent-of "def f()\n-> unit = pass\n" 2) 2)))

(ert-deftest agl-ind-wrapped-body-equals-indents-under-its-signature ()
  (should (= (agl-ind--indent-of "def f() -> unit\n= pass\n" 2) 2)))

(ert-deftest agl-ind-wrapped-binder-equals-indents-under-its-binder ()
  (should (= (agl-ind--indent-of "let total\n= 1 + 2\n" 2) 2)))

(ert-deftest agl-ind-wrapped-branch-arrow-indents-under-its-pattern ()
  (should (= (agl-ind--indent-of "case x of\n  | Pass\n=> ok\n" 3) 4)))

(ert-deftest agl-ind-second-continuation-aligns-with-the-first ()
  ;; `-> unit' already wraps the signature, so the `=' continues that same
  ;; logical line rather than nesting one level deeper again.
  (should (= (agl-ind--indent-of "def f()\n    -> unit\n= pass\n" 3) 4)))

(ert-deftest agl-ind-continuation-symbol-is-not-a-longer-operator ()
  ;; `==' and `->>' are single operator tokens, not the `=' and `->'
  ;; continuation markers, so those lines fall back to the carry-over rule.
  (should (= (agl-ind--indent-of "let ok = a\n== b\n" 2) 0))
  (should (= (agl-ind--indent-of "let ok = a\n->> b\n" 2) 0)))

;; --- Scope regions ---

(ert-deftest agl-ind-body-after-scope-header-is-indented ()
  (should (= (agl-ind--indent-of "scope A\ndef f() -> int = 1\n" 2) 2)))

(ert-deftest agl-ind-scope-closer-aligns-with-its-header ()
  (should (= (agl-ind--indent-of "scope A\n  def f() -> int = 1\nend A\n" 3) 0)))

(ert-deftest agl-ind-scope-closer-skips-a-deeper-declaration-body ()
  ;; The region's last item is a `record' whose fields are deeper still;
  ;; the `end' closes the region, not that record.
  (should (= (agl-ind--indent-of "scope A\n  record R\n    x: int\nend A\n" 4) 0)))

(ert-deftest agl-ind-nested-scope-closer-aligns-with-its-own-header ()
  (should (= (agl-ind--indent-of
              "scope A\n  scope B\n    def f() -> int = 1\n  end B\n" 4)
             2)))

(ert-deftest agl-ind-outer-scope-closer-skips-a-closed-region ()
  (should (= (agl-ind--indent-of
              "scope A\n  scope B\n    def f() -> int = 1\n  end B\nend A\n" 5)
             0)))

(ert-deftest agl-ind-scope-closer-without-a-header-falls-back-to-column-zero ()
  (should (= (agl-ind--indent-of "def f() -> int =\n  1\nend A\n" 3) 0)))

(ert-deftest agl-ind-scope-closer-is-not-a-longer-identifier ()
  ;; `endpoint' is one AgL identifier, not the region closer.
  (should (= (agl-ind--indent-of "scope A\n  let endpoint = 1\n  endpoint\n" 3) 2)))

(ert-deftest agl-ind-indented-region-round-trips ()
  (let ((text (concat "scope A\n"
                      "  record R(x: int)\n"
                      "\n"
                      "  scope B\n"
                      "    def f() -> int = 1\n"
                      "  end B\n"
                      "end A\n")))
    (should (equal (agl-ind--reindented text) text))))

;; --- Bracket continuation ---

(ert-deftest agl-ind-open-paren-aligns-with-content ()
  (should (= (agl-ind--indent-of "let v = f(a,\nb)\n" 2) 10)))

(ert-deftest agl-ind-open-bracket-aligns-with-content ()
  (should (= (agl-ind--indent-of "let xs = [1,\n2]\n" 2) 10)))

(ert-deftest agl-ind-open-brace-aligns-with-content ()
  (should (= (agl-ind--indent-of "let d = { a: 1,\nb: 2 }\n" 2) 10)))

(ert-deftest agl-ind-bracket-opened-at-end-of-line-indents-one-level ()
  (should (= (agl-ind--indent-of "let xs = [\n1,\n]\n" 2) 2)))

(ert-deftest agl-ind-region-dedenting-its-last-line-terminates ()
  ;; Re-indenting a line to a shorter column shrinks the buffer, so a region
  ;; walk bounded by the original end position would never reach it again.
  (should (equal (agl-ind--reindented "def f() -> int =\n  let a = 1\n   let b = 2\n")
                 "def f() -> int =\n  let a = 1\n  let b = 2\n")))

(ert-deftest agl-ind-region-dedenting-an-inner-line-indents-the-rest ()
  ;; The lines after the shortened one still have to be visited.
  (should (equal (agl-ind--reindented
                  "def f() -> int =\n  let a = 1\n   let b = 2\n   let c = 3\n")
                 "def f() -> int =\n  let a = 1\n  let b = 2\n  let c = 3\n")))

(ert-deftest agl-ind-region-keeps-a-body-indented-by-more-than-one-level ()
  ;; A block body takes its level from its own first line, so a four-column
  ;; body under a two-column header is well-formatted and must survive.
  (let ((text (concat "program def main() -> unit =\n"
                      "  case mode of\n"
                      "    | 1 =>\n"
                      "        let empty = \"\"\n"
                      "        print(empty)\n")))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-region-keeps-else-aligned-with-its-if ()
  ;; Moving only one of the two would leave the `else' orphaned, which is the
  ;; damage a level model derived arithmetically from the offset does.
  (let ((text (concat "program def main() -> unit =\n"
                      "  case first of\n"
                      "    | Failed(reason) =>\n"
                      "        if fatal => print reason\n"
                      "        else => print reason\n")))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-region-still-corrects-an-illegal-column ()
  ;; Leniency about deeper bodies must not become leniency about a column that
  ;; belongs to no enclosing block.
  (should (equal (agl-ind--reindented "def f() -> int =\n  let a = 1\n let b = 2\n")
                 "def f() -> int =\n  let a = 1\n  let b = 2\n")))

(ert-deftest agl-ind-region-keeps-guard-markers-aligned-under-an-inline-marker ()
  ;; The markers of an inline guard line up under the first one, which sits
  ;; wherever `if | \=' put it rather than at a multiple of the offset.
  (let ((text (concat "def classify(v: int) -> text =\n"
                      "  if | v < 1 => \"below\"\n"
                      "     | v > 9 => \"above\"\n"
                      "     | else => \"inside\"\n")))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-region-keeps-a-guard-marker-at-its-own-column ()
  ;; Any column deeper than the header opens the branch, so a continuation
  ;; marker that does not line up with the inline one is still well-formed.
  (let ((text (concat "def banner() -> text =\n"
                      "  if | decorate => \"a\"\n"
                      "    | else => \"b\"\n")))
    (should (equal (agl-ind--reindented text) text))))

;; --- Verbatim regions are never re-indented ---

(ert-deftest agl-ind-raw-tail-payload-is-untouched ()
  (let ((text "exec$\n    echo one\n      echo two\n"))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-triple-quoted-template-is-untouched ()
  (let ((text "let doc = \"\"\"\n   ragged\n     lines\n\"\"\"\n"))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-line-after-raw-tail-block-returns-to-code-level ()
  (should (= (agl-ind--indent-of "exec$\n    echo one\nlet after = 1\n" 3) 0)))

;; --- TAB cycling ---

(ert-deftest agl-ind-tab-cycles-to-shallower-levels ()
  (with-temp-buffer
    (agl-mode)
    (insert "def f() -> unit =\n  if\n    | a => 1\n")
    (goto-char (point-max))
    (agl-indent-line)
    (let ((first (current-indentation)))
      (agl-indent-line t)
      (should-not (= (current-indentation) first)))))

(ert-deftest agl-ind-tab-cycle-reaches-column-zero ()
  (with-temp-buffer
    (agl-mode)
    (insert "def f() -> unit =\n    print \"a\"\n")
    (goto-char (point-max))
    (let ((seen nil))
      (agl-indent-line)
      (push (current-indentation) seen)
      (dotimes (_ 6) (agl-indent-line t) (push (current-indentation) seen))
      (should (memq 0 seen)))))

;; --- Electric re-indent ---

(ert-deftest agl-ind-electric-realigns-completed-marker ()
  (with-temp-buffer
    (agl-mode)
    (insert "try\n  risky()\n")
    (insert "    catch")
    (agl-indent-post-self-insert)
    (should (= (current-indentation) 0))))

(ert-deftest agl-ind-electric-ignores-mid-line-word ()
  (with-temp-buffer
    (agl-mode)
    (insert "def f() -> unit =\n  print catch")
    (let ((before (current-indentation)))
      (agl-indent-post-self-insert)
      (should (= (current-indentation) before)))))

;; --- A well-formatted file is a fixed point ---

(ert-deftest agl-ind-formatted-source-round-trips ()
  (let ((text (concat "import std/prelude\n"
                      "\n"
                      "record Point\n"
                      "  var x: int\n"
                      "  y: int\n"
                      "\n"
                      "def describe(p: Point) -> text =\n"
                      "  if\n"
                      "    | p.x == 0 => \"origin\"\n"
                      "    | else => \"point\"\n"
                      "\n"
                      "program def main() -> unit =\n"
                      "  let p = Point(x = 0, y = 0)\n"
                      "  p.x := 1\n"
                      "  print describe(p)\n")))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-pipe-operator-is-not-a-branch-marker ()
  ;; `|>' is one OP_NAME, so a line starting with it continues an
  ;; expression rather than opening a branch.
  (should (= (agl-ind--indent-of "let a = 1\n  let b = 2\n|> g\n" 3) 2)))

(ert-deftest agl-ind-bare-pipe-is-still-a-branch-marker ()
  (should (= (agl-ind--indent-of "let v = if\n| a => 1\n" 2) 2)))

(ert-deftest agl-ind-opener-with-a-trailing-comment-still-opens-a-block ()
  ;; A trailing comment is not code, so it must not stop the line from
  ;; opening its block.
  (should (= (agl-ind--indent-of "def f() =  # note\nx\n" 2) 2))
  (should (= (agl-ind--indent-of "if n > 0 =>  # note\nx\n" 2) 2)))

(ert-deftest agl-ind-comment-only-line-is-skipped-for-layout ()
  (should (= (agl-ind--indent-of "def f() -> unit =\n  print \"a\"\n# note\nprint \"b\"\n" 4) 2)))

;; --- Module-level declarations return to their own level ---

(ert-deftest agl-ind-declaration-returns-to-the-module-root ()
  ;; A `def' is a module item: the grammar never admits one inside a block,
  ;; so it belongs at the root rather than at the body level carried over
  ;; from the function above it.
  (should (= (agl-ind--indent-of "def f() -> unit =\n  print \"a\"\n\ndef g() -> int = 1\n" 4) 0))
  (should (= (agl-ind--indent-of "def f() -> unit =\n  if\n    | a => 1\n\nrecord R\n" 5) 0)))

(ert-deftest agl-ind-declaration-returns-to-its-scope-region ()
  (should (= (agl-ind--indent-of
              "scope A\n  def f() -> unit =\n    print \"a\"\n  def g() -> int = 1\n" 4)
             2)))

(ert-deftest agl-ind-first-declaration-of-a-region-indents-under-its-header ()
  (should (= (agl-ind--indent-of "scope A\n  scope B\nrecord R\n" 3) 4)))

(ert-deftest agl-ind-declaration-after-a-closed-region-returns-to-its-level ()
  (should (= (agl-ind--indent-of
              "scope A\n  scope B\n    def f() -> int = 1\n  end B\n  def g() -> int = 2\n" 5)
             2)))

(ert-deftest agl-ind-statement-keyword-is-not-a-declaration ()
  ;; `print' and `let' are block items; they carry the body's level over
  ;; rather than returning to the module root.
  (should (= (agl-ind--indent-of "def f() -> unit =\n  print \"a\"\n\nprint \"b\"\n" 4) 2))
  (should (= (agl-ind--indent-of "def f() -> unit =\n  print \"a\"\n\nlet b = 1\n" 4) 2)))

;; --- Declarations that carry no body ---

(ert-deftest agl-ind-extern-declaration-does-not-open-a-block ()
  ;; An `extern def' is implemented by its Python companion, so no block
  ;; follows it.
  (should (= (agl-ind--indent-of
              "extern def size(x: array[int]) -> int\n@extern-name(\"first_option\")\n" 2)
             0)))

(ert-deftest agl-ind-builtin-declaration-does-not-open-a-block ()
  ;; A `builtin' on its own line prefixes the declaration under it, and a
  ;; `builtin def' is implemented by the host.
  (should (= (agl-ind--indent-of "builtin\nenum Agent\n" 2) 0))
  (should (= (agl-ind--indent-of "builtin def ask(prompt: text) -> text\nlet x = 1\n" 2) 0)))

(ert-deftest agl-ind-program-modifier-line-does-not-open-a-block ()
  ;; The grammar lets `program' sit on the line above its `def'.
  (should (= (agl-ind--indent-of "program\ndef main() -> unit = pass\n" 2) 0)))

(ert-deftest agl-ind-type-declaration-with-an-inline-body-does-not-open-a-block ()
  ;; A parenthesized field list and an inline member list are the whole
  ;; declaration; only the bodiless header form opens a block.
  (should (= (agl-ind--indent-of "record P(x: int, y: int)\nlet p = 1\n" 2) 0))
  (should (= (agl-ind--indent-of "exception Boom extends Exception()\nlet e = 1\n" 2) 0))
  (should (= (agl-ind--indent-of "enum Flag | On | Off\nlet f = 1\n" 2) 0)))

(ert-deftest agl-ind-type-declaration-header-still-opens-a-block ()
  (should (= (agl-ind--indent-of "record Point\nx: int\n" 2) 2))
  (should (= (agl-ind--indent-of "record Box[T]\nvalue: T\n" 2) 2))
  (should (= (agl-ind--indent-of "exception Boom extends Exception\nreason: text\n" 2) 2))
  (should (= (agl-ind--indent-of "builtin record ExecResult\ncode: int\n" 2) 2))
  (should (= (agl-ind--indent-of "enum Option[T] =\n| None\n" 2) 2)))

;; --- Bracket continuation ---

(ert-deftest agl-ind-closing-bracket-aligns-with-its-opener ()
  ;; The closer ends the logical line the opener began, so it returns to
  ;; that line's level instead of sitting at the content column.
  (should (= (agl-ind--indent-of "let v = f(a,\n          b,\n)\n" 3) 0))
  (should (= (agl-ind--indent-of "def f() -> unit =\n  let v = g(a,\n)\n" 3) 2)))

(ert-deftest agl-ind-body-after-a-wrapped-signature-indents-from-its-start ()
  ;; The signature's last line sits at the argument column, but the item it
  ;; belongs to began at column zero, so its body is one level from there.
  (should (= (agl-ind--indent-of
              "program def main(\n    verbose: bool = false) -> unit =\n  print verbose\n" 3)
             2)))

(ert-deftest agl-ind-statement-after-a-wrapped-call-returns-to-its-level ()
  (should (= (agl-ind--indent-of "let v = f(a,\n          b)\nlet w = 2\n" 3) 0)))

;; --- Branch markers ---

(ert-deftest agl-ind-pipe-marker-aligns-with-its-sibling ()
  ;; The branch above ends in a suite of its own; the next `|' is that
  ;; branch's sibling, not a branch of something the suite opened.
  (should (= (agl-ind--indent-of
              "def f() -> int =\n  if\n    | a =>\n        1\n    | else => 2\n" 5)
             4)))

;; --- Attributes stand with what they prefix ---

(ert-deftest agl-ind-attribute-returns-to-the-declaration-level ()
  ;; The attribute belongs to the `extern def' under it, so it stands where
  ;; that declaration does rather than at the body level above.
  (should (= (agl-ind--indent-of
              "def f() -> int =\n  try g() catch E as e => 0\n\n@extern-name(\"is_file\")\n" 4)
             0)))

(ert-deftest agl-ind-field-attribute-keeps-the-field-level ()
  ;; Inside a record-like body an attribute prefixes a field, not a
  ;; declaration, so the body's level carries over.
  (should (= (agl-ind--indent-of "record R\n@arg-named a: int\n" 2) 2))
  (should (= (agl-ind--indent-of "exception Boom extends Exception\n  message: text\n@arg-named code: int\n" 3) 2)))

;; --- Word markers find the construct they name ---

(ert-deftest agl-ind-loop-terminator-aligns-past-a-nested-construct ()
  ;; The `case' inside the loop body is not what `until' closes, so the
  ;; search walks out to the `do' rather than stopping at the first line
  ;; indented less than the body.
  (should (= (agl-ind--indent-of
              "def f() -> unit =\n  do\n    case v of\n      | A => ()\n  until done?\n" 5)
             2))
  (should (= (agl-ind--indent-of
              "def f() -> unit =\n  while cond\n    if a =>\n      step()\n  done\n" 5)
             2)))

(ert-deftest agl-ind-catch-aligns-with-its-own-try ()
  (should (= (agl-ind--indent-of
              "try\n  try\n    risky()\n  catch A as e => 1\ncatch B as e => 2\n" 5)
             0)))

(ert-deftest agl-ind-second-catch-aligns-with-the-first ()
  ;; A `try' takes several `catch' clauses, and they stand together.
  (should (= (agl-ind--indent-of
              "def f() -> int =\n  try\n    risky()\n  catch A as e => 1\n  catch B as e => 2\n" 5)
             2)))

(ert-deftest agl-ind-else-aligns-with-an-inline-if ()
  ;; The `if' wrote its consequent inline, so it opens no block, but it is
  ;; still the header the `else' continues.
  (should (= (agl-ind--indent-of
              "def f() -> int =\n  case v of\n    | A =>\n        if a => 1\n        else => 2\n" 5)
             8)))

;; --- Typing ---

(ert-deftest agl-ind-newline-keeps-a-wider-body ()
  ;; A body takes its level from its own first line, so ending that line
  ;; must not pull it back to the one level the engine would have computed.
  (should (equal (agl-ind--typed "def f() -> unit =\n  print \"a\"\nprint \"b\"")
                 "def f() -> unit =\n    print \"a\"\n    print \"b\"")))

(ert-deftest agl-ind-newline-keeps-a-dedented-line ()
  (should (equal (agl-ind--typed "def f() -> unit =\nlet a = 1\n\rlet b = 2\nlet c = 3")
                 "def f() -> unit =\n  let a = 1\nlet b = 2\nlet c = 3")))

(ert-deftest agl-ind-typing-a-declaration-keyword-returns-to-its-level ()
  ;; The space after `def' is the first moment the word can be told from an
  ;; identifier that merely starts with it, and the line is placed then.
  (should (equal (agl-ind--typed "def f() -> unit =\nprint \"a\"\n\ndef ")
                 "def f() -> unit =\n  print \"a\"\n\ndef "))
  (should (equal (agl-ind--typed "def f() -> unit =\nprint \"a\"\n\nrecord ")
                 "def f() -> unit =\n  print \"a\"\n\nrecord ")))

(ert-deftest agl-ind-typing-a-longer-identifier-keeps-the-line-put ()
  ;; `default-agent' begins with the letters of `def' without being it, so
  ;; nothing moves while it is typed.
  (should (equal (agl-ind--typed "def f() -> unit =\nlet a = default-agent")
                 "def f() -> unit =\n  let a = default-agent")))

(provide 'agl-indent-tests)
;;; agl-indent-tests.el ends here
