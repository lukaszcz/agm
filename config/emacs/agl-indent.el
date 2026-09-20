;;; agl-indent.el --- Indentation engine for agl-mode -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; A hand-written indentation engine for AgL, in the python-mode family.
;;
;; AgL uses significant indentation, so no single "correct" column exists
;; after a block ends: the engine computes one target and TAB then cycles
;; through the enclosing levels.  The rules it implements come from the
;; layout section of docs/agl/reference/lexical-structure.md:
;;
;; - Bracket continuation: while a `(', `[', `{', or `%{' interpolation is
;;   open, the logical line continues, and a continuation line aligns with
;;   the bracket's content column — except the line that closes the
;;   bracket, which returns to the level of the line that opened it.
;; - Branch-marker continuation: a line whose first token is `|', `else',
;;   `catch', `until', or `done' continues the enclosing construct and
;;   aligns with the line that opened it, or with the `|' sibling that
;;   already stands at the branch column.
;; - Symbolic continuation: a line whose first token is `->', `=>', or `='
;;   wraps the line above it — a signature's return type, a body's or a
;;   binder's `=', a branch's arrow — and indents one level under the line
;;   it continues, or alongside it when that line is itself such a wrap.
;; - Module-level declarations: the grammar admits a `def', `record',
;;   `enum', `exception', `type', `import', `export', or `scope' only at a
;;   module's root and inside a scope region, never inside a block body, so
;;   such a line returns to the level of the region enclosing it rather
;;   than carrying the body above it over.
;; - Block opening: a line that opens a suite indents its body one
;;   `agl-indent-offset' deeper.  A declaration that carries no body — an
;;   `extern def', a `builtin def', a bare modifier line, a record or enum
;;   whose fields are written inline — opens nothing.
;; - Otherwise the previous logical line's indentation carries over, taken
;;   from where that logical line began rather than from its last physical
;;   line.
;;
;; A line is placed again as soon as typing settles which construct it is:
;; on the `|', `@', or closing bracket that can only start one thing, on the
;; whitespace that tells a declaration keyword from a name beginning with
;; the same letters, on the last letter of a branch marker, and — for a
;; line holding nothing but `builtin', `extern', or `program', which no
;; separator ever follows — on the newline that ends it.
;;
;; `$' verbatim block payloads and multi-line templates are verbatim text,
;; so a line inside one is never re-indented, and backward scans treat
;; those regions as opaque.  Both checks read `syntax-ppss' rather than
;; the buffer text, since the propertize layer is what marks those regions.

;;; Code:

;; `agl-mode.el' requires this file after defining the mode, so requiring it
;; back would be circular; the few helpers used from it are declared instead.
(declare-function agl--ident-boundary-after-p "agl-mode" (pos))
(declare-function agl--ident-boundary-before-p "agl-mode" (pos))
(declare-function agl--operator-name-char-p "agl-mode" (char))
(declare-function agl--standalone-token-start-p "agl-mode" (pos))

(defcustom agl-indent-offset 2
  "Number of columns AgL indents a nested block, matching stdlib style."
  :type 'integer
  :safe #'integerp
  :group 'agl)

;; ---------------------------------------------------------------------------
;; Line predicates
;; ---------------------------------------------------------------------------

(defconst agl--branch-marker-re
  (concat "\\(?:|\\|" (regexp-opt '("else" "catch" "until" "done")) "\\)")
  "Regexp matching a branch marker that continues the enclosing construct.

The markers are `|', `else', `catch', `until', and `done' (see the
layout rules in docs/agl/reference/lexical-structure.md).  A scope
region's `end' also closes what a header opened, but it names the region
it closes rather than continuing the nearest construct, so it is placed
by `agl--scope-closer-indent' instead.")

(defconst agl--marker-or-closer-re
  (concat "\\(?:" agl--branch-marker-re "\\|end\\)")
  "Regexp matching a branch marker or a scope region's closer.

Both change which construct the line being typed belongs to, so both are
worth re-indenting on as soon as the word is complete.")

(defconst agl--continuation-symbol-re
  "\\(?:=>\\|->\\|=\\)"
  "Regexp matching a symbolic token that can only continue the line above.

No AgL construct begins with `->', `=>', or `=', so a line opening with
one wraps the line before it (see the layout rules in
docs/agl/reference/lexical-structure.md).  The two-character spellings
lead the alternation so a maximal match wins.")

(defconst agl--block-opener-symbol-re
  "\\(?:=>\\|->\\|[=:]\\)[ \t]*$"
  "Regexp matching a symbolic suite introducer ending a line's code.

A line ending in `=', `:', `=>', or `->' has its body on the following
lines.  The spelling alone does not settle it: `a->' is a single AgL
identifier and the `=' of `>=' belongs to that comparison operator, so
`agl--block-opener-symbol-p' accepts a match only where it is a whole
operator token.")

(defconst agl--block-opener-keyword-re
  (concat (regexp-opt '("of" "do" "try" "else" "then" "if")) "[ \t]*$")
  "Regexp matching a keyword suite introducer ending a line's code.

As with `agl--block-opener-symbol-re' the spelling must be a whole token:
`registry', `undo', and `motif' end in these letters without being them,
which is what `agl--block-opener-keyword-p' checks.")

(defconst agl--verbatim-block-opener-re "\\$[ \t]*$"
  "Regexp matching a `$' verbatim literal opener that carries no inline payload.

`$' is escaped since it would otherwise be read as the regexp
end-of-line anchor.")

(defconst agl--declaration-keyword-re
  (regexp-opt '("def" "record" "enum" "exception" "type" "extern"
                "builtin" "program" "import" "use" "export" "scope"
                "infixl" "infixr"))
  "Regexp matching a keyword that declares a module item.

The grammar admits these declarations at a module's root and inside a
scope region only, never inside a block body (see the item list in
docs/agl/reference/grammar.md), so such a line returns to the level of
the region enclosing it instead of continuing the body above.  `let' and
`var' are absent: they declare bindings, which a block does admit.")

(defconst agl--declaration-head-re
  "[^][ \t(){},|;=]+"
  "Regexp matching the name a declaration declares.

An AgL name consumes everything that is not whitespace and not a
structural delimiter, and `::' makes a qualified head one token
(`enum Workflow::Status'), so the class subtracts the delimiters rather
than listing what a name admits.")

(defconst agl--modifier-only-re
  (concat "\\`[ \t]*" (regexp-opt '("builtin" "extern" "program")) "[ \t]*\\'")
  "Regexp matching a line carrying nothing but a declaration modifier.

`builtin', `extern', and `program' each prefix a declaration the grammar
lets start on the line below, which is then that modifier's sibling
rather than its body.")

(defconst agl--body-less-function-re
  (concat "\\`[ \t]*" (regexp-opt '("builtin" "extern")) "[ \t]+def[ \t]")
  "Regexp matching a function declaration that never carries a body.

A `builtin def' is implemented by the host and an `extern def' by its
Python companion, so no block follows either.")

(defconst agl--type-declaration-re
  (concat "\\`[ \t]*\\(?:builtin[ \t]+\\)?"
          (regexp-opt '("record" "enum" "exception")) "[ \t]")
  "Regexp matching a record, enum, or exception declaration line.")

(defconst agl--type-header-re
  (concat "\\`[ \t]*\\(?:builtin[ \t]+\\)?"
          (regexp-opt '("record" "enum" "exception"))
          "[ \t]+" agl--declaration-head-re
          "\\(?:[ \t]*\\[[^]]*\\]\\)?"
          "\\(?:[ \t]+extends[ \t]+" agl--declaration-head-re "\\)?"
          "[ \t]*=?[ \t]*\\'")
  "Regexp matching a record, enum, or exception whose body is a block.

These three declarations may write their members inline instead —
`record P(x: int)', `enum Flag | On | Off' — and then own no block at
all, so a header opens one only when nothing but the declared name, its
type parameters, an `extends' base, and the optional `=' is on the
line.")

(defconst agl--block-header-re
  (concat "\\`[ \t]*\\(?:"
          (regexp-opt '("record" "enum" "exception" "scope" "program" "def"
                        "extern" "builtin" "for" "while" "do" "if" "case" "try"))
          "\\)")
  "Regexp matching a line that starts a declaration or compound statement.

Anchored at the line start: one of these keywords further along the line
is an argument or string content (`print(\"do it\")'), not a header.")

(defun agl--line-empty-p ()
  "Return non-nil when the current line has only whitespace on it."
  (save-excursion
    (beginning-of-line)
    (looking-at-p "[ \t]*$")))

(defun agl--line-comment-p ()
  "Return non-nil if the current line's first token is a comment."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (not (eolp))
         (eq (char-after) ?#)
         (save-excursion (nth 4 (syntax-ppss (1+ (point))))))))

(defun agl--line-skippable-p ()
  "Return non-nil if the current line is ignored for layout purposes.

Blank lines and comment-only lines do not participate in layout."
  (or (agl--line-empty-p) (agl--line-comment-p)))

(defun agl--line-code-end ()
  "Return the position where the current line's code ends.

A trailing comment is not code, so it is excluded; the result is the
end of line when the line carries no comment."
  (save-excursion
    (beginning-of-line)
    (let ((limit (line-end-position))
          (found nil))
      (while (and (not found) (< (point) limit))
        (if (and (eq (char-after) ?#)
                 ;; `syntax-ppss' leaves point at its argument, so the scan
                 ;; is wrapped: without it the recorded position lands after
                 ;; the `#' and the comment stays in the code text.
                 (save-excursion (nth 4 (syntax-ppss (1+ (point))))))
            (setq found (point))
          (forward-char 1)))
      (or found limit))))

(defun agl--opaque-line-p ()
  "Return non-nil if the current line lies inside a verbatim region.

A `$' verbatim block payload and a multi-line template are significant
text, so a line whose start is already inside one is never re-indented."
  ;; `syntax-ppss' leaves point at its argument, so the scan is wrapped:
  ;; a predicate that silently moved point would corrupt every caller.
  (save-excursion
    (let* ((bol (line-beginning-position))
           (probe (min (point-max) (1+ bol))))
      (and (or (nth 3 (syntax-ppss bol))
               (nth 3 (syntax-ppss probe)))
           t))))

(defvar agl--crossed-verbatim-region nil
  "Set by `agl--goto-previous-code-line' when it skipped a verbatim region.

A `$' verbatim block payload IS its opener's block, so a line following
the payload returns to the opener's own level instead of indenting under it.")

(defun agl--goto-previous-code-line ()
  "Move to the previous line that participates in layout.

Skips blank and comment-only lines, and skips over verbatim regions so a
`$' verbatim payload or template body never acts as the previous line.
Return non-nil when such a line was found."
  (let ((found nil))
    (setq agl--crossed-verbatim-region nil)
    (while (and (not found) (zerop (forward-line -1)))
      (cond
       ((agl--line-skippable-p) nil)
       ((agl--opaque-line-p)
        ;; Move to where the region began and keep scanning: its interior is
        ;; verbatim text, and the opener's own line is the code line wanted.
        ;; Stop here only when that line carries code before the region.
        (let* ((bol (line-beginning-position))
               (start (or (nth 8 (syntax-ppss (min (point-max) (1+ bol))))
                          (nth 8 (syntax-ppss bol)))))
          (when start
            (goto-char start)
            (setq agl--crossed-verbatim-region t)
            (setq found (save-excursion
                          (skip-chars-backward " \t")
                          (not (bolp))))
            (beginning-of-line))))
       (t (setq found t))))
    found))

;; ---------------------------------------------------------------------------
;; Indentation computation
;; ---------------------------------------------------------------------------

(defun agl--closing-bracket-line-p ()
  "Return non-nil if the current line opens with a closing bracket."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (memq (char-after) '(?\) ?\] ?\})) t)))

(defun agl--enclosing-bracket-column ()
  "Return the content column of the innermost open bracket, or nil.

While a bracket is open the logical line continues, so a continuation
line aligns just past that bracket.  A line that opens with the closing
bracket ends that logical line instead of continuing it, so it returns
to the level of the line the bracket was opened on."
  (let* ((state (save-excursion (syntax-ppss (line-beginning-position))))
         (open (nth 1 state)))
    (when open
      (save-excursion
        (cond
         ((agl--closing-bracket-line-p)
          (goto-char open)
          (current-indentation))
         (t
          (goto-char open)
          (forward-char 1)
          (skip-chars-forward " \t")
          (if (eolp)
              (+ (progn (goto-char open) (current-indentation)) agl-indent-offset)
            (current-column))))))))

(defun agl--logical-line-indentation ()
  "Return the indentation of the line the current line\='s item began on.

A bracket continues one logical line across newlines, so a line written
inside one has no level of its own: what places the item, and the block
it may open, is the column its first line sits at."
  (let* ((state (save-excursion (syntax-ppss (line-beginning-position))))
         (open (car (nth 9 state))))
    (if open
        (save-excursion (goto-char open) (current-indentation))
      (current-indentation))))

(defun agl--branch-marker-line-p ()
  "Return non-nil if the current line begins with a branch marker."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (looking-at agl--branch-marker-re)
         (if (eq (char-after) ?|)
             ;; A branch `|' stands alone.  Operator characters after it
             ;; make one `OP_NAME' instead — `|>' is a user infix operator,
             ;; so such a line continues an expression rather than opening
             ;; a branch.
             (let ((next (char-after (1+ (point)))))
               (or (null next) (memq next '(?\s ?\t ?\n))))
           ;; A word marker must not be the prefix of a longer AgL
           ;; identifier (`done-with' is one name).
           (agl--ident-boundary-after-p (match-end 0))))))

(defun agl--continuation-symbol-line-p ()
  "Return non-nil if the current line begins with a symbolic continuation.

The spelling alone does not settle it: an operator name is a maximal run
of operator characters, so the `==' opening a line is one comparison
token rather than the `=' marker, and `->>' is one user operator rather
than `->'."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (looking-at agl--continuation-symbol-re)
         (not (agl--operator-name-char-p (char-after (match-end 0)))))))

(defun agl--continuation-symbol-indent ()
  "Return the column the current line\='s symbolic continuation should sit at.

The marker wraps the previous code line, so it indents one level under
it — unless that line is itself a wrap of the same logical line, in which
case the two align."
  (save-excursion
    (beginning-of-line)
    (if (not (agl--goto-previous-code-line))
        0
      (let ((previous (current-indentation)))
        (if (or agl--crossed-verbatim-region (agl--continuation-symbol-line-p))
            previous
          (+ previous agl-indent-offset))))))

(defun agl--first-word-p (word)
  "Return non-nil when WORD is the current line\='s first whole token."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (looking-at-p (regexp-quote word))
         (agl--ident-boundary-after-p (+ (point) (length word))))))

(defun agl--scope-closer-line-p ()
  "Return non-nil if the current line closes a scope region."
  (agl--first-word-p "end"))

(defun agl--declaration-line-p ()
  "Return non-nil if the current line\='s first token declares a module item."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (looking-at agl--declaration-keyword-re)
         (agl--ident-boundary-after-p (match-end 0)))))

(defun agl--attribute-line-p ()
  "Return non-nil if the current line\='s first token is an attribute."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (eq (char-after) ?@)))

(defun agl--type-declaration-line-p ()
  "Return non-nil if the current line declares a record, enum, or exception."
  (and (string-match-p agl--type-declaration-re
                       (buffer-substring-no-properties
                        (line-beginning-position) (agl--line-code-end)))
       t))

(defun agl--field-attribute-p ()
  "Return non-nil when the current line\='s attribute prefixes a field.

A `record\=', `enum\=', or `exception\=' body holds fields rather than
declarations, so an attribute written in one prefixes a field.  The body
is recognized from its header: the nearest line above the attribute that
is indented less than the line it follows, or that line itself when the
attribute opens the body.

The walk stops at that line whether or not it declares a type.  Going on
past it would leave the body altogether and reach the declarations
around it, and any record or exception among those would claim an
attribute that prefixes something else entirely."
  (save-excursion
    (beginning-of-line)
    (when (agl--goto-previous-code-line)
      (or (agl--type-declaration-line-p)
          (let ((body (current-indentation))
                (header nil)
                (found nil))
            (while (and (not found) (agl--goto-previous-code-line))
              (when (< (current-indentation) body)
                (setq found t
                      header (agl--type-declaration-line-p))))
            header)))))

(defun agl--attribute-indent ()
  "Return the column this line\='s attribute should sit at.

An attribute stands immediately above what it prefixes and takes that
line\='s column, so it is placed as the declaration below it will be —
except inside a record, enum, or exception body, where what follows is a
field and the body\='s level carries over."
  (if (agl--field-attribute-p)
      (agl--carried-over-indent)
    (agl--declaration-indent)))

(defun agl--enclosing-region-column (declaration)
  "Return the column the region enclosing the current line places, or nil.

The region is found structurally, by walking back over the preceding
code lines and letting each `end\=' skip the `scope\=' it already closed:
the enclosing indentation cannot name it, since a region\='s last item may
itself be a `record\=' or `case\=' header whose own body is deeper.

With DECLARATION nil the answer is the region header\='s own column, which
is where its `end\=' belongs.  With DECLARATION non-nil it is where the
region\='s items belong, so a header reports one level in and the walk
also stops at a declaration already written at that level."
  (save-excursion
    (beginning-of-line)
    (let ((depth 0)
          (column nil))
      (while (and (null column) (agl--goto-previous-code-line))
        (cond ((agl--scope-closer-line-p) (setq depth (1+ depth)))
              ((agl--first-word-p "scope")
               (if (> depth 0)
                   (setq depth (1- depth))
                 (setq column (if declaration
                                  (+ (current-indentation) agl-indent-offset)
                                (current-indentation)))))
              ((and declaration (zerop depth) (agl--declaration-line-p))
               (setq column (current-indentation)))))
      column)))

(defun agl--scope-closer-indent ()
  "Return the column this line\='s `end\=' should sit at.

An `end\=' closes the nearest scope region still open above it and stands
at that region header\='s own column."
  (or (agl--enclosing-region-column nil) 0))

(defun agl--declaration-indent ()
  "Return the column this line\='s module-level declaration should sit at.

A declaration belongs to the module root or to a scope region, never to
a block body, so it returns to the level of the region enclosing it —
named by the region\='s header or by a declaration already written there."
  (or (agl--enclosing-region-column t) 0))

(defun agl--pipe-marker-line-p ()
  "Return non-nil if the current line begins with a `|\=' branch marker."
  (and (agl--branch-marker-line-p)
       (save-excursion
         (beginning-of-line)
         (skip-chars-forward " \t")
         (eq (char-after) ?|))))

(defconst agl--marker-opener-alist
  '(("else" "if")
    ("catch" "try")
    ("until" "for" "while" "do")
    ("done" "for" "while" "do"))
  "The header keywords each word marker continues.

`|\=' is absent: it introduces a branch of whatever construct encloses it
— an `if\=', a `case\=', an `enum\=' — so its owner is found by position
rather than by name.")

(defun agl--line-first-word ()
  "Return the current line\='s first word, or nil when it starts otherwise.

The word must be a whole AgL token: `done-with\=' is one identifier that
merely begins with the letters of a marker."
  (save-excursion
    (beginning-of-line)
    (skip-chars-forward " \t")
    (when (looking-at "[a-z]+")
      (let ((word (match-string-no-properties 0)))
        (and (agl--ident-boundary-after-p (match-end 0)) word)))))

(defun agl--word-marker-owner (marker)
  "Return the column of the construct MARKER continues, or nil.

MARKER is `else\=', `catch\=', `until\=', or `done\=', each of which names the
headers it can continue, so the owner is found by walking out through
the lines that enclose this one — those at a strictly smaller
indentation than any seen so far — to the nearest such header.  A
construct nested in the body passed on the way is never one of them,
which is what the enclosing walk buys over stopping at the first
shallower line.

A clause of the same word met on the way is a sibling — one `try\=' takes
several `catch\=' clauses — and stands where this one belongs, but only
until a header is found further out: the sibling may instead belong to a
construct nested inside the one being continued."
  (let ((openers (cdr (assoc marker agl--marker-opener-alist)))
        (limit nil)
        (sibling nil)
        (column nil))
    (save-excursion
      (beginning-of-line)
      (while (and (null column) (agl--goto-previous-code-line))
        (let ((indent (current-indentation)))
          (when (or (null limit) (< indent limit))
            (setq limit indent)
            (let ((word (agl--line-first-word)))
              (cond ((member word openers) (setq column indent))
                    ((and (equal word marker) (null sibling))
                     (setq sibling indent))))))))
    (or column sibling)))

(defun agl--positional-owner (pipe)
  "Return (INDENT . OPENS-BLOCK) for the construct a branch marker continues.

The marker's own column is whatever the user has typed so far, so it is
not used.  The search starts from the previous code line: when that line
opens a suite it IS the construct's header, and otherwise the header is
the nearest preceding line indented less than it.

PIPE is non-nil when the marker being placed is a `|\='.  A `|\=' whose
search lands on another `|\=' has found a sibling branch rather than a
header, and siblings share a column: the branch that came first already
stands where this one belongs, whether it opened a suite of its own or
not."
  (save-excursion
    (when (agl--goto-previous-code-line)
      (cond
       ((and pipe (agl--pipe-marker-line-p)) (cons (current-indentation) nil))
       ((agl--opens-block-p) (cons (current-indentation) t))
       (t
        (let ((previous (current-indentation))
              (owner nil))
          (while (and (not owner) (agl--goto-previous-code-line))
            (when (< (current-indentation) previous)
              (setq owner (if (and pipe (agl--pipe-marker-line-p))
                              (cons (current-indentation) nil)
                            (cons (current-indentation) (agl--opens-block-p))))))
          (or owner (cons 0 nil))))))))

(defun agl--branch-marker-indent ()
  "Return the column the current line's branch marker should sit at.

A `|' introduces a branch inside the construct's body, so it indents one
level under a header that opens a suite.  The word markers `else',
`catch', `until', and `done' close or continue the construct itself and
align with its header, which they name: only when no such header stands
above them is the header guessed from position instead."
  (let ((pipe (agl--pipe-marker-line-p)))
    (or (and (not pipe) (agl--word-marker-owner (agl--line-first-word)))
        (let* ((owner (agl--positional-owner pipe))
               (indent (or (car owner) 0))
               (opens (cdr owner)))
          (if (and pipe opens) (+ indent agl-indent-offset) indent)))))

(defun agl--block-opener-symbol-p (code start)
  "Return non-nil when CODE ends with a symbolic suite introducer.

CODE is the current line\='s code text taken from buffer position START, so
a match\='s index in CODE is also its position in the buffer.  The
introducer must be an operator token in its own right
\(`agl--standalone-token-start-p'): `a->' is one identifier whose `->'
never lexes apart, the `=' ending `>=' or `!=' continues that operator
instead of assigning, and a `=' or `:' appearing as ordinary text inside
a still-open `$' verbatim payload on the same line is not code at all."
  (and (string-match agl--block-opener-symbol-re code)
       (agl--standalone-token-start-p (+ start (match-beginning 0)))))

(defun agl--block-opener-keyword-p (code start)
  "Return non-nil when CODE ends with a keyword suite introducer.

CODE and START are as in `agl--block-opener-symbol-p'.  An identifier
consumes the keyword\='s letters when they merely end a longer name, so the
match counts only where it begins a token (`registry' is not `try')."
  (and (string-match agl--block-opener-keyword-re code)
       (agl--ident-boundary-before-p (+ start (match-beginning 0)))))

(defun agl--verbatim-opener-p (code start)
  "Return non-nil when CODE ends with a `$' verbatim opener carrying no payload.

CODE and START are as in `agl--block-opener-symbol-p'; such an opener owns
the indented block that follows it.  The `$' must begin a token in its own
right (`agl--standalone-token-start-p'): inside an identifier or an
operator-name run (`ask$', `<$>', `|$') it is an ordinary constituent
rather than an opener, and a `$' written as ordinary text inside a
still-open verbatim payload earlier on the same line (`exec $ echo
price $') is payload content, not a second opener."
  (and (string-match agl--verbatim-block-opener-re code)
       (agl--standalone-token-start-p (+ start (match-beginning 0)))))

(defun agl--block-header-p (code start)
  "Return non-nil when CODE is a declaration or compound-statement header.

CODE and START are as in `agl--block-opener-symbol-p'.  A header owns a
block only when its body is not written inline on the same line, and
only when the declaration has a body at all: a bare modifier, a
`builtin def', an `extern def', and a record or enum whose members are
written inline all own nothing that follows them."
  (and (string-match agl--block-header-re code)
       (agl--ident-boundary-after-p (+ start (match-end 0)))
       (not (string-match-p "=[ \t]*[^ \t]" code))
       (not (string-match-p agl--modifier-only-re code))
       (not (string-match-p agl--body-less-function-re code))
       (or (not (string-match-p agl--type-declaration-re code))
           (string-match-p agl--type-header-re code))))

(defun agl--opens-block-p ()
  "Return non-nil if the current line opens a nested block.

A line opens a block when its code ends with a suite introducer, or when
it is a declaration or compound-statement header with no inline body.
The code text is taken unshortened from the line\='s start, so each
predicate can map a match back to its buffer position and settle there
whether the spelling it found is a whole AgL token."
  (let* ((start (line-beginning-position))
         (code (buffer-substring-no-properties start (agl--line-code-end))))
    (and (string-match-p "[^ \t]" code)
         (or (agl--block-opener-symbol-p code start)
             (agl--block-opener-keyword-p code start)
             (agl--verbatim-opener-p code start)
             (agl--block-header-p code start)))))

(defun agl--carried-over-indent ()
  "Return the column carried over from the line above the current one.

The previous logical line sets the level: its own when it opens nothing,
and one `agl-indent-offset\=' deeper when it opens a block.  A `$' verbatim
payload IS its opener\='s block, so crossing one returns to the opener\='s
level rather than nesting under it."
  (save-excursion
    (beginning-of-line)
    (if (not (agl--goto-previous-code-line))
        0
      (let ((previous (agl--logical-line-indentation))
            (crossed agl--crossed-verbatim-region))
        (if (and (agl--opens-block-p) (not crossed))
            (+ previous agl-indent-offset)
          previous)))))

(defun agl-calculate-indent ()
  "Return the column `agl-indent-line' should indent the current line to."
  (save-excursion
    (beginning-of-line)
    (cond
     ;; Inside a bracket the logical line continues.
     ((agl--enclosing-bracket-column))
     ;; A scope closer aligns with the region header it closes.
     ((agl--scope-closer-line-p) (agl--scope-closer-indent))
     ;; A branch marker aligns with the construct it continues.
     ((agl--branch-marker-line-p) (agl--branch-marker-indent))
     ;; A symbolic continuation wraps the line above rather than opening a block.
     ((agl--continuation-symbol-line-p) (agl--continuation-symbol-indent))
     ;; A module-level declaration returns to the region that encloses it.
     ((agl--declaration-line-p) (agl--declaration-indent))
     ;; An attribute is placed as the line it prefixes will be.
     ((agl--attribute-line-p) (agl--attribute-indent))
     (t (agl--carried-over-indent)))))

(defun agl--enclosing-levels ()
  "Return the columns of the blocks enclosing the current line, deepest first.

These are the strictly decreasing indentations of the preceding code
lines: each is a column some enclosing line actually sits at, which is
what makes it a column this line may legally return to.  A layout
language takes each body\='s level from that body\='s first line, so a body
may be indented by more than `agl-indent-offset\=' and the enclosing
levels cannot be derived arithmetically from it."
  (save-excursion
    (beginning-of-line)
    (let ((levels nil)
          (deepest nil))
      (while (and (or (null deepest) (> deepest 0))
                  (agl--goto-previous-code-line))
        (let ((column (current-indentation)))
          (when (or (null deepest) (< column deepest))
            (setq deepest column)
            (push column levels))))
      (nreverse levels))))

(defun agl--indent-levels ()
  "Return the candidate indentation columns for the current line.

The computed target comes first, then each enclosing level down to
column zero: an indentation-sensitive language cannot know which level a
line belongs to once a block has ended, so TAB offers them in turn."
  (let ((levels (list (agl-calculate-indent))))
    (dolist (level (agl--enclosing-levels))
      (unless (memq level levels) (setq levels (append levels (list level)))))
    (unless (memq 0 levels) (setq levels (append levels (list 0))))
    levels))

(defun agl--indentation-settled-p ()
  "Return non-nil when the current line\='s indentation is already legal.

A line may sit at any enclosing level (`agl--indent-levels\='), and a line
that starts a nested body may sit at any column deeper than that body\='s
header: the first line is what chooses the level, so re-indenting it to
the one computed target would rewrite well-formatted source.  That holds
for a block body and equally for a `|\=' branch, whose markers commonly
align under an inline first marker (`if | a => x\=').  The terminators
`else\=', `catch\=', `until\=', `done\=' and `end\=' continue or close the construct
itself rather than opening a body, so they stay subject to the levels
above."
  (let ((column (current-indentation)))
    (or (memq column (agl--indent-levels))
        (save-excursion
          (beginning-of-line)
          (and (agl--goto-previous-code-line)
               (not agl--crossed-verbatim-region)
               (agl--opens-block-p)
               (> column (current-indentation))))
        (save-excursion
          (beginning-of-line)
          (and (agl--pipe-marker-line-p)
               (let ((owner (agl--positional-owner t)))
                 (and owner (> column (car owner)))))))))

;; ---------------------------------------------------------------------------
;; Commands
;; ---------------------------------------------------------------------------

(defun agl-indent-line (&optional previous)
  "Indent the current line as AgL code.

With PREVIOUS non-nil (a repeated TAB), cycle to the next candidate
level instead of re-applying the computed one.  A line inside a `$'
verbatim payload or a multi-line template is left untouched: its text is
verbatim."
  (interactive)
  (unless (agl--opaque-line-p)
    (let* ((levels (agl--indent-levels))
           (current (current-indentation))
           (target (if (not previous)
                       (car levels)
                     (or (cadr (member current levels)) (car levels))))
           (offset (- (current-column) current)))
      (indent-line-to target)
      (when (> offset 0) (forward-char (min offset (- (line-end-position) (point))))))))

(defun agl-indent-line-function ()
  "`indent-line-function' for `agl-mode', cycling on a repeated TAB."
  (agl-indent-line (and (eq last-command 'indent-for-tab-command)
                        (eq this-command 'indent-for-tab-command))))

(defun agl--marker-just-completed-p ()
  "Return non-nil when the last key completed a branch marker or `end\='.

The marker is matched here rather than through a predicate, so the check
never depends on `match-data' surviving another call."
  (let ((end (point)))
    (save-excursion
      (beginning-of-line)
      (skip-chars-forward " \t")
      (and (looking-at agl--marker-or-closer-re)
           (= end (match-end 0))
           (or (eq (char-after) ?|)
               (agl--ident-boundary-after-p (match-end 0)))))))

(defun agl--declaration-just-opened-p ()
  "Return non-nil when the last key ended a line\='s declaration keyword.

The trigger is the whitespace after the keyword rather than its last
letter: an AgL name may begin with those letters and go on
(`default-agent' is one name), and the separator is the first character
that tells the keyword from such a name."
  (and (memq last-command-event '(?\s ?\t))
       (string-match-p (concat "\\`[ \t]*" agl--declaration-keyword-re "[ \t]\\'")
                       (buffer-substring-no-properties
                        (line-beginning-position) (point)))))

(defun agl--attribute-just-opened-p ()
  "Return non-nil when the last key opened an attribute on this line.

`@\=' begins an attribute and begins nothing else, so unlike a declaration
keyword it needs no separator to tell it from a name: the keystroke
itself settles which construct the line is."
  (and (eq last-command-event ?@)
       (string-match-p "\\`[ \t]*@\\'"
                       (buffer-substring-no-properties
                        (line-beginning-position) (point)))))

(defun agl--closer-just-typed-p ()
  "Return non-nil when the last key closed a bracket at the line\='s head.

A closer returns to the line that opened its bracket, and like `@\=' it
names its own construct: nothing else starts a line with it."
  (and (memq last-command-event '(?\) ?\] ?}))
       (agl--closing-bracket-line-p)
       (= (point) (save-excursion
                    (beginning-of-line)
                    (skip-chars-forward " \t")
                    (1+ (point))))))

(defun agl--modifier-only-line-p ()
  "Return non-nil when the current line carries a lone declaration modifier."
  (string-match-p agl--modifier-only-re
                  (buffer-substring-no-properties
                   (line-beginning-position) (agl--line-code-end))))

(defun agl--place-preceding-modifier-line ()
  "Place the lone modifier line the newline just ended, if that is one.

`electric-indent-inhibit\=' keeps RET from re-indenting the line it ends,
because a body line has several legal columns and the one already there
is the user\='s.  A line holding nothing but `builtin\=', `extern\=', or
`program\=' is not such a line: the grammar settles its level, and being
whole words with nothing after them they are never followed by the
separator that places a declaration keyword as it is typed.

Returns non-nil when a line moved."
  (save-excursion
    (when (and (zerop (forward-line -1))
               (not (agl--opaque-line-p))
               (agl--modifier-only-line-p))
      (let ((target (agl-calculate-indent)))
        (unless (= target (current-indentation))
          (indent-line-to target)
          t)))))

(defun agl-indent-post-self-insert ()
  "Re-indent the current line once its first token settles where it belongs.

Typing `|\=' or `@\=' at the start of a line, finishing one of the words
`else\=', `catch\=', `until\=', `done\=', or `end\=' there, or separating a
declaration keyword such as `def\=' from the name after it all decide
which construct the line belongs to, so the line is re-indented
immediately.  A newline settles the lone modifier line it ends, which no
keystroke of its own ever will, and the line it opens then follows."
  (when (and (eq major-mode 'agl-mode)
             (not (agl--opaque-line-p)))
    (cond ((eq last-command-event ?\n)
           (when (agl--place-preceding-modifier-line)
             (agl-indent-line)))
          ((or (agl--marker-just-completed-p)
               (agl--declaration-just-opened-p)
               (agl--attribute-just-opened-p)
               (agl--closer-just-typed-p))
           (agl-indent-line)))))

(defun agl-indent-region (start end)
  "Indent each line between START and END as AgL code.

A line whose indentation is already one of the valid levels is left
alone.  Significant indentation means a line that ends a block has
several legal columns, and a block body may be indented by more than one
level, so re-indenting every line to one computed target would rewrite
well-formatted source; `agl-indent-line' still applies that target when
the user asks for it on a single line.

END is tracked with a marker because re-indenting a line changes how
long it is: shortening one moves every later position, and a fixed END
would then sit past the end of the buffer, leaving the walk unable to
reach it."
  (let ((limit (copy-marker end)))
    (save-excursion
      (goto-char start)
      (beginning-of-line)
      (while (< (point) limit)
        (unless (or (agl--opaque-line-p) (agl--line-empty-p)
                    (agl--indentation-settled-p))
          (indent-line-to (agl-calculate-indent)))
        (forward-line 1)))
    (set-marker limit nil)))

(provide 'agl-indent)
;;; agl-indent.el ends here
