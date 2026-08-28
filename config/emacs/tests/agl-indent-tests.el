;;; agl-indent-tests.el --- ERT tests for agl-mode indentation -*- lexical-binding: t; -*-

;;; Commentary:

;; Scenario tables for the AgL indentation engine: block opening and
;; closing, branch-marker alignment for every marker, bracket
;; continuation, TAB cycling, electric re-indent, and the immunity of
;; raw-tail payloads and multi-line templates.  See the layout rules in
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

;; --- Bracket continuation ---

(ert-deftest agl-ind-open-paren-aligns-with-content ()
  (should (= (agl-ind--indent-of "let v = f(a,\nb)\n" 2) 10)))

(ert-deftest agl-ind-open-bracket-aligns-with-content ()
  (should (= (agl-ind--indent-of "let xs = [1,\n2]\n" 2) 10)))

(ert-deftest agl-ind-open-brace-aligns-with-content ()
  (should (= (agl-ind--indent-of "let d = { a: 1,\nb: 2 }\n" 2) 10)))

(ert-deftest agl-ind-bracket-opened-at-end-of-line-indents-one-level ()
  (should (= (agl-ind--indent-of "let xs = [\n1,\n]\n" 2) 2)))

;; --- Verbatim regions are never re-indented ---

(ert-deftest agl-ind-raw-tail-payload-is-untouched ()
  (let ((text "exec!\n    echo one\n      echo two\n"))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-triple-quoted-template-is-untouched ()
  (let ((text "let doc = \"\"\"\n   ragged\n     lines\n\"\"\"\n"))
    (should (equal (agl-ind--reindented text) text))))

(ert-deftest agl-ind-line-after-raw-tail-block-returns-to-code-level ()
  (should (= (agl-ind--indent-of "exec!\n    echo one\nlet after = 1\n" 3) 0)))

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
  (let ((text (concat "import std/core\n"
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

(provide 'agl-indent-tests)
;;; agl-indent-tests.el ends here
