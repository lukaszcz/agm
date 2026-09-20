;;; agl-syntax-tests.el --- ERT tests for agl-mode syntax/lexing -*- lexical-binding: t; -*-

;;; Commentary:

;; Tests the context-sensitive `syntax-propertize' layer of `agl-mode':
;; identifier atomicity (quotes, '#', and operator characters are
;; identifier-continuation characters), the four string-template forms,
;; comments, `$' verbatim text literal payloads, and the consequences
;; (forward-sexp, comment-dwim, syntax-ppss) that fall out of getting the
;; syntax-table properties right.  See
;; docs/agl/reference/lexical-structure.md for the authoritative rules.

;;; Code:

(require 'ert)
(require 'agl-mode)

(defmacro agl-test--with-buffer (text &rest body)
  "Evaluate BODY in a temporary `agl-mode' buffer propertized from TEXT.

Insert TEXT, run `syntax-propertize' over the whole buffer, then
evaluate BODY with point at `point-min'."
  (declare (indent 1))
  `(with-temp-buffer
     (agl-mode)
     (insert ,text)
     (syntax-propertize (point-max))
     (goto-char (point-min))
     ,@body))

(defmacro agl-test--with-edited-buffer (text edit-fn &rest body)
  "Evaluate BODY in a temporary `agl-mode' buffer edited after propertizing.

Insert TEXT, propertize it, mutate it with EDIT-FN, propertize it
again, then evaluate BODY with point at `point-min'.

EDIT-FN is a function of no arguments, called with point at
`point-min'; its insertions and deletions go through the normal
buffer-change machinery, so they invalidate `syntax-propertize''s
cache exactly as a real interactive edit would.  This exercises the
`agl--propertize-extend-region' path that `agl-test--with-buffer'
\(a single whole-buffer propertize\) never visits."
  (declare (indent 2))
  `(with-temp-buffer
     (agl-mode)
     (insert ,text)
     (syntax-propertize (point-max))
     (goto-char (point-min))
     (funcall ,edit-fn)
     (syntax-propertize (point-max))
     (goto-char (point-min))
     ,@body))

(defmacro agl-test--with-chunked-buffer (text chunk-size &rest body)
  "Evaluate BODY in a temporary `agl-mode' buffer propertized in chunks.

Insert TEXT, then call `syntax-propertize' repeatedly with
`syntax-propertize-chunk-size' bound to CHUNK-SIZE, in CHUNK-SIZE-sized
steps, instead of once for the whole buffer, then evaluate BODY with
point at `point-min'.

`syntax-propertize' always processes at least
`syntax-propertize-chunk-size' characters per call regardless of the
position it is asked to reach (see `syntax-propertize' in
`syntax.el'), so binding it down to CHUNK-SIZE is what actually forces
successive calls to resume from a boundary in the middle of a
construct -- the way `jit-lock-mode' really propertizes a buffer in
small windows as it becomes visible.  A single whole-buffer propertize
call never exercises `agl--propertize-extend-region' this way."
  (declare (indent 2))
  `(with-temp-buffer
     (agl-mode)
     (insert ,text)
     (let ((syntax-propertize-chunk-size ,chunk-size)
           (pos (point-min)))
       (while (< pos (point-max))
         (syntax-propertize pos)
         (setq pos (min (point-max) (+ pos ,chunk-size)))))
     (goto-char (point-min))
     ,@body))

(defun agl-test--pos-after (needle)
  "Return the buffer position right after the first occurrence of NEEDLE."
  (save-excursion
    (goto-char (point-min))
    (search-forward needle)
    (point)))

(defun agl-test--in-string-p (pos)
  "Return non-nil if POS is inside a string/template per `syntax-ppss'."
  (nth 3 (syntax-ppss pos)))

(defun agl-test--in-comment-p (pos)
  "Return non-nil if POS is inside a comment per `syntax-ppss'."
  (nth 4 (syntax-ppss pos)))

;; --- Identifiers: quotes, '#', and operator characters are constituents ---

(ert-deftest agl-syntax-quote-inside-identifier-not-string ()
  (agl-test--with-buffer "let x = foo\"bar\nlet y = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "foo\"bar")))
    (should-not (agl-test--in-string-p (point-max)))))

(ert-deftest agl-syntax-quote-token-start-opens-string ()
  (agl-test--with-buffer "let s = \"hi\"\n"
    (should (agl-test--in-string-p (agl-test--pos-after "\"h")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "\"hi\"")))))

(ert-deftest agl-syntax-hash-inside-identifier-not-comment ()
  (agl-test--with-buffer "let x = foo#bar\nlet y = 1\n"
    (should-not (agl-test--in-comment-p (agl-test--pos-after "foo#bar")))
    (should-not (agl-test--in-comment-p (point-max)))))

(ert-deftest agl-syntax-hash-token-start-is-comment ()
  (agl-test--with-buffer "# note\nlet x = 1  # note\nlet y = 2\n"
    (should (agl-test--in-comment-p (agl-test--pos-after "# no")))
    (should (agl-test--in-comment-p (agl-test--pos-after "1  # no")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "let y")))))

(ert-deftest agl-syntax-identifier-operator-chars-are-symbol-constituents ()
  (agl-test--with-buffer "let ask-prompt = valid?\n"
    (goto-char (agl-test--pos-after "let "))
    (should (equal (thing-at-point 'symbol) "ask-prompt"))))

(ert-deftest agl-syntax-arrow-identifier-is-one-token ()
  (agl-test--with-buffer "let a->b = 1\n"
    (goto-char (agl-test--pos-after "let "))
    (should (equal (thing-at-point 'symbol) "a->b"))))

(ert-deftest agl-syntax-unicode-identifier-start ()
  (agl-test--with-buffer "let café = 1\n"
    (goto-char (agl-test--pos-after "let "))
    (should (equal (thing-at-point 'symbol) "café"))))

;; --- Templates: the four forms ---

(ert-deftest agl-syntax-double-quote-single-line ()
  (agl-test--with-buffer "let a = \"hi\"\n"
    (should (agl-test--in-string-p (agl-test--pos-after "\"h")))))

(ert-deftest agl-syntax-single-quote-single-line ()
  (agl-test--with-buffer "let a = 'hi'\n"
    (should (agl-test--in-string-p (agl-test--pos-after "'h")))))

(ert-deftest agl-syntax-triple-double-quote-with-single-quotes-and-hash ()
  (agl-test--with-buffer "let a = \"\"\"it's a #test\"\"\"\n"
    (should (agl-test--in-string-p (agl-test--pos-after "it's a #test")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "it's a #test")))
    (should-not (agl-test--in-string-p (point-max)))))

(ert-deftest agl-syntax-triple-single-quote-with-double-quotes ()
  (agl-test--with-buffer "let a = '''hello \"world\"'''\n"
    (should (agl-test--in-string-p (agl-test--pos-after "hello \"world\"")))
    (should-not (agl-test--in-string-p (point-max)))))

(ert-deftest agl-syntax-triple-quote-multiline ()
  (agl-test--with-buffer
      (concat "let a = \"\"\"line one\n"
              "line two # not a comment\n"
              "line three\"\"\"\n"
              "let b = 1\n")
    (should (agl-test--in-string-p (agl-test--pos-after "line two")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "line two # not a comment")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let b")))))

(ert-deftest agl-syntax-unterminated-single-line-does-not-swallow ()
  (agl-test--with-buffer "let a = \"unterminated\nlet b = 2\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "let b = 2")))))

;; --- `$' verbatim text literals ---

(ert-deftest agl-syntax-verbatim-exec-inline-hash-is-payload ()
  (agl-test--with-buffer "exec $ ls -la # not a comment\nlet x = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "ls -la # not a comment")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "# not a comment")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-verbatim-ask-inline ()
  (agl-test--with-buffer "ask $ Summarize %{topic} please\nlet y = 2\n"
    (should (agl-test--in-string-p (agl-test--pos-after "Summarize")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let y")))))

(ert-deftest agl-syntax-verbatim-dotted-projection ()
  (agl-test--with-buffer "receiver.ask $ do the thing\nlet z = 3\n"
    (should (agl-test--in-string-p (agl-test--pos-after "do the thing")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let z")))))

(ert-deftest agl-syntax-verbatim-type-args-are-on-the-callee ()
  ;; Type arguments are written on the callee, before the `$', not after it.
  (agl-test--with-buffer "ask::[Review] $ prompt text\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "::[Review")))
    (should (agl-test--in-string-p (agl-test--pos-after "prompt text")))))

(ert-deftest agl-syntax-verbatim-token-start-at-line-start ()
  ;; A bare `$' opens the literal wherever a new token starts, including
  ;; with no callee before it at all.
  (agl-test--with-buffer "$ ls -la\nlet x = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "ls -la")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-verbatim-dollar-inside-identifier-is-not-an-opener ()
  ;; `ask$ x' is not `ask $ x': `$' continues the identifier `ask$', so this
  ;; line opens no verbatim literal at all.
  (agl-test--with-buffer "ask$ x\nlet y = 2\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "ask$ x")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let y")))))

(ert-deftest agl-syntax-verbatim-dollar-inside-operator-is-not-an-opener ()
  ;; `<$>' is a single operator-name token: its `$' does not begin a token.
  (agl-test--with-buffer "let v = a <$> b\nlet w = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "<$> b")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let w")))))

(ert-deftest agl-syntax-verbatim-dollar-continuing-an-operator-run-is-not-an-opener ()
  ;; `=', `|', and `/' each terminate an identifier scan (they are
  ;; `IDENT_STOP' members), but a `$' that immediately follows one of them
  ;; still merges into a single operator name (`<|$', `=$', `!=$', `|$',
  ;; `/$'), so none of these opens a payload either.
  (agl-test--with-buffer "print <|$ zz\nlet a = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "<|$ zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let a"))))
  (agl-test--with-buffer "x =$ zz\nlet b = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "=$ zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let b"))))
  (agl-test--with-buffer "x !=$ zz\nlet c = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "!=$ zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let c"))))
  (agl-test--with-buffer "a |$ zz\nlet d = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "|$ zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let d"))))
  (agl-test--with-buffer "a /$ zz\nlet e = 1\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "/$ zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let e")))))

(ert-deftest agl-syntax-verbatim-dollar-with-a-space-before-it-still-opens ()
  ;; The canonical spelling always separates the `$' from its callee (or, as
  ;; here, from an operator) with a space, which keeps it a token of its own
  ;; even when the character right before the space is an operator character.
  (agl-test--with-buffer "print <| $ zz\nlet a = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let a"))))
  (agl-test--with-buffer "let x = $ y\nlet b = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "y")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let b")))))

(ert-deftest agl-syntax-verbatim-dollar-after-a-closed-string-opens ()
  ;; A closing quote ends a string/template atomically, so whatever follows
  ;; it -- `$' included -- begins a fresh token, unlike a quote merely
  ;; swallowed into a longer identifier (`foo"bar', not a string at all).
  (agl-test--with-buffer "\"a\"$ zz\nlet x = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-verbatim-dollar-after-a-bare-number-opens ()
  ;; A number token ends at its last digit, so a `$' immediately following one
  ;; still begins a fresh token -- unlike a digit run that is part of a
  ;; longer identifier (`a1$', one name, `$' included).
  (agl-test--with-buffer "1$ zz\nlet x = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "zz")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-verbatim-block-payload ()
  (agl-test--with-buffer
      (concat "ask $\n"
              "  Summarize the report.\n"
              "\n"
              "  Mention \"quotes\" and # not-a-comment.\n"
              "let after = 1\n")
    (should (agl-test--in-string-p (agl-test--pos-after "Summarize the report.")))
    (should (agl-test--in-string-p (agl-test--pos-after "Mention \"quotes\" and # not-a-comment.")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "# not-a-comment")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let after")))))

(ert-deftest agl-syntax-verbatim-block-ends-at-opener-indentation ()
  (agl-test--with-buffer
      (concat "if cond\n"
              "  ask $\n"
              "    Explain this.\n"
              "  let x = 1\n")
    (should (agl-test--in-string-p (agl-test--pos-after "Explain this.")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-verbatim-payload-parens-are-inert ()
  (agl-test--with-buffer "exec $ echo (unbalanced\nlet x = [1, 2]\n"
    (should (agl-test--in-string-p (agl-test--pos-after "(unbalanced")))
    (goto-char (agl-test--pos-after "let x = "))
    (forward-sexp 1)
    (should (eq (char-before) ?\]))))

;; --- Consequences: forward-sexp, comment-dwim, extend-region ---

(ert-deftest agl-syntax-forward-sexp-over-brackets ()
  (agl-test--with-buffer "let xs = [1, 2, foo(bar, baz)]\n"
    (goto-char (agl-test--pos-after "= "))
    (forward-sexp 1)
    (should (eq (char-before) ?\]))))

(ert-deftest agl-syntax-comment-dwim-inserts-line-comment ()
  (agl-test--with-buffer "let x = 1\nlet y = 2\n"
    (goto-char (agl-test--pos-after "let x = 1"))
    (comment-dwim nil)
    (should (looking-back "# " (line-beginning-position)))))

(ert-deftest agl-syntax-verbatim-block-survives-edit-inside-payload ()
  "Editing inside a `$' verbatim block payload must not corrupt its syntax.

The buggy extend-region only backed up to `agl-multiline''s start
inside the block, never to the opener line, so the opener was never
rescanned and the payload was relexed as code."
  (agl-test--with-edited-buffer
      (concat "ask $\n"
              "  line one\n"
              "  line two\n"
              "let x = 1\n")
      (lambda ()
        (goto-char (agl-test--pos-after "line two"))
        (insert "X"))
    (should (agl-test--in-string-p (agl-test--pos-after "line twoX")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "line twoX")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-triple-quoted-survives-edit-inside-string ()
  "Editing inside a triple-quoted template must not corrupt its syntax."
  (agl-test--with-edited-buffer
      "let a = \"\"\"line one\nline two\nline three\"\"\"\n"
      (lambda ()
        (goto-char (agl-test--pos-after "line two"))
        (insert "X"))
    (should (agl-test--in-string-p (agl-test--pos-after "line twoX")))
    (should-not (agl-test--in-string-p (point-max)))))

(ert-deftest agl-syntax-verbatim-block-survives-chunked-propertize ()
  "Chunked propertizing must still see the whole `$' verbatim block as a string.

Chunk boundaries (as `jit-lock-mode' produces, bounded by
`syntax-propertize-chunk-size') land inside the block and inside the
opener line."
  (agl-test--with-chunked-buffer
      (concat "ask $\n"
              "  line one\n"
              "  line two\n"
              "  line three\n"
              "let x = 1\n")
      5
    (should (agl-test--in-string-p (agl-test--pos-after "line two")))
    (should-not (agl-test--in-comment-p (agl-test--pos-after "line two")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-triple-quoted-survives-chunked-propertize ()
  "Chunked propertizing must still see the whole triple-quoted template as a string."
  (agl-test--with-chunked-buffer
      (concat "let a = \"\"\"line one\n"
              "line two\n"
              "line three\"\"\"\n"
              "let b = 1\n")
      5
    (should (agl-test--in-string-p (agl-test--pos-after "line two")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let b")))))

;; --- Keyword inventory constants (data only; not wired to font-lock yet) ---
;;
;; Each constant documents its own canonical Python source (see the
;; docstrings in agl-mode.el: `src/agm/agl/keywords.py',
;; `src/agm/agl/lexer/tokens.py'). A test that only asserts these
;; hand-written constants contain hand-written strings cannot detect drift
;; from those Python sources, so none is included here.

;; --- `$' verbatim payload backslashes: owned as text, not escape syntax ---

(ert-deftest agl-syntax-verbatim-payload-trailing-backslash-does-not-escape-fence ()
  (agl-test--with-buffer "exec $ echo foo\\\nlet x = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "foo\\")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

(ert-deftest agl-syntax-verbatim-block-payload-trailing-backslash-does-not-escape-fence ()
  "A payload line ending in `\\' must not escape the block's close fence.

Otherwise the block stays open and swallows the following code."
  (agl-test--with-buffer
      (concat "ask $\n"
              "  line one\n"
              "  line two\\\n"
              "let x = 1\n")
    (should (agl-test--in-string-p (agl-test--pos-after "line two\\")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let x")))))

;; --- `$' verbatim literals are only recognized at bracket depth zero ---

(ert-deftest agl-syntax-verbatim-dollar-inside-call-args-is-not-an-opener ()
  (agl-test--with-buffer "let y = f($, 1)\nlet z = 2\n"
    (should-not (agl-test--in-string-p (agl-test--pos-after "$, 1")))
    (goto-char (agl-test--pos-after "let y = f"))
    (forward-sexp 1)
    (should (eq (char-before) ?\)))
    (should (= 0 (car (syntax-ppss (agl-test--pos-after "let z")))))))

;; --- `%{...}' interpolation holes do not break sexp balance ---

(ert-deftest agl-syntax-template-hole-with-nested-call-and-quotes-balances-parens ()
  (agl-test--with-buffer "let a = \"hello %{f(\"x\")} world\"\n"
    (should (= 0 (car (syntax-ppss (point-max)))))))

(ert-deftest agl-syntax-template-escaped-percent-brace-is-literal ()
  (agl-test--with-buffer "let a = \"cost is \\%{100}\"\nlet b = 1\n"
    (should (agl-test--in-string-p (agl-test--pos-after "cost is")))
    (should-not (agl-test--in-string-p (agl-test--pos-after "let b")))))

;; --- Degenerate `$' verbatim payload lengths ---

(ert-deftest agl-syntax-verbatim-single-char-inline-payload-at-eof-is-string ()
  (agl-test--with-buffer "exec $ x"
    (should (agl-test--in-string-p (point-max)))))

(ert-deftest agl-syntax-verbatim-inline-payload-at-eof-keeps-last-char-as-content ()
  (agl-test--with-buffer "exec $ ab"
    (should (agl-test--in-string-p (agl-test--pos-after "ab")))))

(ert-deftest agl-syntax-verbatim-block-payload-at-eof-without-newline-is-string ()
  (agl-test--with-buffer (concat "ask $\n" "  line one\n" "  line two")
    (should (agl-test--in-string-p (point-max)))))

(ert-deftest agl-syntax-verbatim-block-drops-trailing-blank-lines ()
  ;; The scanner drops the blank lines after a block payload's last content
  ;; line, so they are not part of the verbatim region.
  (agl-test--with-buffer "exec $\n  a\n\n"
    (should-not (nth 3 (syntax-ppss (1- (point-max)))))))

(ert-deftest agl-syntax-verbatim-block-keeps-its-content ()
  (agl-test--with-buffer "exec $\n  a\n  b\nlet after = 1\n"
    (goto-char (point-min))
    (search-forward "  b")
    (should (nth 3 (syntax-ppss (1- (point)))))
    (goto-char (point-min))
    (search-forward "let after")
    (should-not (nth 3 (syntax-ppss (1- (point)))))))

(provide 'agl-syntax-tests)
;;; agl-syntax-tests.el ends here
