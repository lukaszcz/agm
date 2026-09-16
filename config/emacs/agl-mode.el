;;; agl-mode.el --- Major mode for AgL source files -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; A major mode for AgL, the statically typed workflow language implemented
;; by AGM (Agent Project Management).  This file provides the package
;; skeleton, the context-sensitive lexing layer, structural font-lock, and
;; declaration navigation (imenu, `beginning-of-defun'/`end-of-defun').
;;
;; The lexing layer is a syntax table plus a `syntax-propertize-function'
;; that make Emacs agree with AgL's lexical rules (see
;; docs/agl/reference/lexical-structure.md), in particular:
;;
;; - An identifier starts with a Unicode letter or `_' and then greedily
;;   consumes every character that is not whitespace and not one of the
;;   structural delimiters `( ) [ ] { } : , . | ; / @ ='.  Quotes, `#', and
;;   the operator characters `- ? ! + * < >' are therefore identifier
;;   constituents: `foo"bar', `foo#bar', `a+b', and `ask-prompt' are each one
;;   identifier.  A quote or `#' only starts a template or comment when it is
;;   not in the middle of an identifier scan.
;; - Four string-template forms -- `"..."', `'...'', `"""..."""',
;;   `'''...''''  -- with `%{expr}' interpolation.
;; - Raw tails (`exec$'/`ask$', bare or after a `.' projection, with an
;;   optional byte-adjacent `::[T]' type argument) own a verbatim payload:
;;   the rest of the line, or a following indented block.
;;
;; Key bindings: `C-c C-c' runs the file (`agm exec'), `C-c C-k' checks it
;; (`agm check'), `C-c C-z' opens the inferior REPL, and `C-c C-r' /
;; `C-c C-b' send the region or buffer to it.
;;
;; Font-lock is structural only : capitalization is semantically
;; meaningless in AgL, so faces derive from declaration and annotation
;; positions, never from spelling; constructor use-sites in expressions
;; stay unfaced.  Indentation, flymake, and the exec/REPL integrations
;; live in the sibling files.  See docs/agl/reference/lexical-structure.md
;; for the authoritative lexical rules.

;;; Code:

(defgroup agl nil
  "Major mode for editing AgL source files."
  :group 'languages
  :prefix "agl-")

;; ---------------------------------------------------------------------------
;; Keyword inventories, wired into font-lock further below.
;;
;; Canonical sources, kept in lockstep with this file and
;; config/micro/agl.yaml -- see the cross-reference comment in
;; src/agm/agl/keywords.py.
;; ---------------------------------------------------------------------------

(defconst agl-keywords
  '("record" "enum" "type" "program" "def" "fn" "let" "var"
    "for" "while" "do" "until" "done" "if" "else" "case" "of" "try" "catch"
    "raise" "return" "break" "continue" "exception" "extends" "builtin"
    "extern" "as" "as?"
    "true" "false" "null" "infixl" "infixr" "prio")
  "Reserved AgL keywords.

Canonical source: `src/agm/agl/keywords.py' (the `KEYWORDS' frozenset).")

(defconst agl-constant-keywords
  '("true" "false" "null")
  "AgL literal keywords, a subset of `agl-keywords'.")

(defconst agl-operator-keywords
  '("and" "or" "not" "is" "in" "to" "downto" "step" "with")
  "AgL operator words, a subset of `agl-soft-keywords'.

Promoted to operators in operator position only, but painted as
keywords wherever they appear: lexical highlighting cannot tell that
position from a member name, and the operator reading is the common
one.")

(defconst agl-soft-keywords
  '("import" "use" "export" "hiding" "scope" "end"
    "and" "or" "not" "is" "in" "to" "downto" "step" "with")
  "AgL soft (contextually promoted) keywords.

Ordinary identifiers outside their promotion window.  The operator
words are promoted in operator position only, so they name members
and fields everywhere else.  Canonical source:
`src/agm/agl/keywords.py' (the `SOFT_KEYWORDS' frozenset), whose
promoted token types are `src/agm/agl/lexer/tokens.py''s
`SOFT_KEYWORD_TOKENS'.")

(defconst agl-contextual-builtins
  '("print" "render" "exec" "ask" "ask-request"
    "copy" "shallow-copy" "resource" "resource-dir")
  "The AgL builtin call names, highlighted by spelling.

These lex as ordinary NAME tokens; the stdlib declares them as
`builtin def' across `packages/stdlib/src/*.agl'.  Canonical source:
`src/agm/agl/scope/symbols.py' (the `BUILTIN_CALL_NAMES' mapping),
mirrored here in full -- update this list with that one.

`regexp-opt' folds shared prefixes into a greedy trie, so `ask-request'
and `resource-dir' win over `ask' and `resource' at the same position,
and `agl--search-ident-forward' rejects any match that does not end on
an AgL identifier boundary -- `-' continues a name, so `resource-path'
and `copy-of' stay unfaced.")

(defconst agl-raw-tail-names
  '("exec$" "ask$")
  "Raw-tail opener spellings.  Canonical source:
`src/agm/raw_tail_catalog.py' (`RAW_TAIL_NAMES').")

(defconst agl-primitive-type-names
  '("unit" "text" "json" "bool" "int" "decimal" "array" "dict")
  "AgL primitive type-annotation names.

Ordinary identifiers outside type-annotation positions.  Canonical
source: `docs/agl/reference/lexical-structure.md' (type-annotation
keywords).")

;; ---------------------------------------------------------------------------
;; Identifier scanning.
;;
;; Mirrors `IDENT_STOP' in `src/agm/util/ident.py': an identifier starts
;; with a Unicode letter or `_' and then consumes every character that is
;; not whitespace and not one of the structural delimiters below.
;; ---------------------------------------------------------------------------

(defconst agl--ident-start-re "[[:alpha:]_]"
  "Regexp matching the first character of an AgL identifier.")

(defconst agl--ident-continue-skip "^][(){}:,.|;/@=\t\n\r "
  "`skip-chars-forward' spec matching AgL identifier-continuation characters.

The complement of `IDENT_STOP' in `src/agm/util/ident.py'.")

(defconst agl--ident-stop-chars (substring agl--ident-continue-skip 1)
  "The characters that terminate an AgL identifier.

The bracket-expression body of `agl--ident-continue-skip', with its
leading `^' negation marker stripped.")

(defconst agl--ident-stop-char-re (concat "[" agl--ident-stop-chars "]")
  "Regexp matching one AgL identifier-terminating character.")

(defconst agl--ident-stop-char-list (append agl--ident-stop-chars nil)
  "The characters that terminate an AgL identifier, as a char list.

Same character set as `agl--ident-stop-char-re', but usable with
`memq' -- allocation-free and immune to the `match-data' clobbering a
regexp match would risk (see `agl--ident-boundary-after-p').")

(defconst agl--ident-continue-char-re (concat "[^" agl--ident-stop-chars "]")
  "Regexp matching one AgL identifier-continuation character.")

(defconst agl--name-re (concat agl--ident-start-re agl--ident-continue-char-re "*")
  "Regexp matching one complete AgL `NAME' token.")

;; ---------------------------------------------------------------------------
;; Identifier-boundary-safe searching.
;;
;; Keywords, contextual builtins, and other fixed spellings must not match
;; inside a larger AgL identifier.  Every character but the `IDENT_STOP'
;; delimiters continues a name (see `agl--ident-continue-skip', which spells
;; that set as its complement) -- operator characters, quotes and `#'
;; included -- so `ask-prompt' and `a+and+b' are each one name, and a naive
;; `\\b'-anchored regexp would wrongly light up `ask' or `and' inside them.
;; `agl--search-ident-forward' is the shared primitive every font-lock
;; matcher below is built from.
;; ---------------------------------------------------------------------------

(defun agl--ident-boundary-before-p (pos)
  "Return non-nil unless POS lands inside a larger AgL identifier.

Walks backward from POS over identifier-continuation characters; POS is
embedded in a larger identifier only when that backward run is
non-empty AND starts with a letter or `_'.  A run starting with any
other continuation character -- an operator character, say -- never
began as an identifier (`-3' lexes as `MINUS' then `INT', not one run),
so it does not block a match at POS; this is what lets a keyword or
number match right after an unspaced arrow or unary minus."
  (save-excursion
    (goto-char pos)
    (let ((run-end (point)))
      (skip-chars-backward agl--ident-continue-skip)
      (or (= (point) run-end)
          (not (looking-at agl--ident-start-re))))))

(defun agl--ident-boundary-after-p (pos)
  "Return non-nil unless POS abuts an AgL identifier-continuation character.

See `agl--ident-continue-skip' for the character set checked."
  (or (= pos (point-max))
      (memq (char-after pos) agl--ident-stop-char-list)))

(defun agl--search-ident-forward (regexp limit &optional pred)
  "Search forward for REGEXP up to LIMIT, accepting only boundary-safe matches.

AgL is case-sensitive, so the search binds `case-fold-search' to nil
regardless of the buffer's default -- otherwise, e.g., `DEF loud()'
would be mistaken for a `def' declaration.  A match is accepted only at
AgL identifier boundaries (see `agl--ident-boundary-before-p' and
`agl--ident-boundary-after-p').  When PRED is given, an otherwise-accepted
match is accepted only if `(funcall PRED (match-beginning 0))' is also
non-nil.  Return non-nil on success, with point left at the end of the
accepted match and `match-data' set to describe it -- the boundary
checks and PRED are free to call `looking-at'/`string-match' internally
\(they do), which would otherwise clobber the global match data this
function's own match relies on, so it is saved and restored around
them."
  (let ((case-fold-search nil) found)
    (while (and (not found) (re-search-forward regexp limit t))
      (let ((saved (match-data)) (mbeg (match-beginning 0)) (mend (match-end 0)))
        (if (and (agl--ident-boundary-before-p mbeg)
                 (agl--ident-boundary-after-p mend)
                 (progn (set-match-data saved)
                        (or (null pred) (funcall pred mbeg))))
            (progn (set-match-data saved) (setq found t))
          (goto-char (1+ mbeg)))))
    found))

;; ---------------------------------------------------------------------------
;; syntax-propertize helpers.
;;
;; A single left-to-right pass assigns `syntax-table' text properties.
;; Identifiers are consumed atomically (via `agl--ident-continue-skip'), so
;; quotes and `#' inside them are naturally inert -- they are never visited
;; as standalone characters by the dispatch loop below.  Multi-line
;; constructs (triple-quoted templates, raw-tail blocks) are additionally
;; marked with the internal `agl-multiline' text property so that
;; `agl--propertize-extend-region' can always resume scanning from a safe
;; boundary.  A raw-tail payload is further marked with the internal
;; `agl-raw-tail-payload' text property, distinguishing it from a template
;; region (both carry the same generic-string-fence `|' syntax, but only a
;; raw-tail payload uses raw-tail backslash-escape semantics -- see
;; `agl--escaped-interpolation-open-p').
;; ---------------------------------------------------------------------------

(defun agl--blank-line-p ()
  "Return non-nil if the line at point contain only whitespace."
  (save-excursion
    (beginning-of-line)
    (looking-at "[ \t]*$")))

(defun agl--propertize-raw-block (opener-indent opener-line-start)
  "Propertize a raw-tail block payload as a generic string.

Point must be at the end of the opener's line (only whitespace follows
the opener on that line).  The payload is every following line more
indented than OPENER-INDENT (blank lines included), up to, but not
including, the first non-blank line at or below OPENER-INDENT.
OPENER-LINE-START is the beginning of the opener's own line: the
`agl-multiline' property is applied from there, not just from the
payload, so `agl--propertize-extend-region' can always back up to the
opener and never resumes scanning from inside the payload."
  (forward-line 1)
  (let ((block-start (point)))
    (while (and (not (eobp))
                (or (agl--blank-line-p) (> (current-indentation) opener-indent)))
      (forward-line 1))
    ;; Trailing blank lines are not payload: the scanner drops the blank
    ;; lines that follow the last content line, so the closing fence goes
    ;; there rather than at the end of the run.
    (let* ((block-end (save-excursion
                        (let ((limit (point)))
                          (goto-char limit)
                          (while (and (> (point) block-start)
                                      (save-excursion
                                        (forward-line -1)
                                        (agl--blank-line-p)))
                            (forward-line -1))
                          (point))))
           (has-content (> block-end block-start)))
      (when has-content
        (put-text-property block-start (1+ block-start)
                            'syntax-table (string-to-syntax "|"))
        (put-text-property block-start block-end 'agl-raw-tail-payload t)
        (agl--propertize-raw-interpolation-holes block-start block-end)
        (if (and (eq (char-before block-end) ?\n)
                 (> (1- block-end) block-start))
            (progn
              (put-text-property (1- block-end) block-end
                                  'syntax-table (string-to-syntax "|"))
              (agl--propertize-backslashes-as-punctuation (1+ block-start) (1- block-end)))
          ;; No room for (or no) a distinct closing fence character --
          ;; the block runs to end of buffer without a trailing
          ;; newline.  Fence only the opener and leave the string
          ;; unterminated, matching triple-quote handling.
          (agl--propertize-backslashes-as-punctuation (1+ block-start) block-end))
        (put-text-property opener-line-start block-end 'agl-multiline t))
      (goto-char block-end))))

(defun agl--propertize-raw-inline ()
  "Propertize an inline raw-tail payload as a generic string.

The payload is the rest of the current line, starting at point."
  (let* ((payload-start (point))
         (line-end (line-end-position))
         (has-trailing-newline (< line-end (point-max)))
         (region-end (if has-trailing-newline (1+ line-end) (point-max))))
    (put-text-property payload-start (1+ payload-start)
                        'syntax-table (string-to-syntax "|"))
    (put-text-property payload-start region-end 'agl-raw-tail-payload t)
    (agl--propertize-raw-interpolation-holes payload-start region-end)
    (if (and has-trailing-newline (> (1- region-end) payload-start))
        (progn
          (put-text-property (1- region-end) region-end
                              'syntax-table (string-to-syntax "|"))
          (agl--propertize-backslashes-as-punctuation (1+ payload-start) (1- region-end)))
      ;; No trailing newline to serve as a synthetic close fence -- the
      ;; payload runs to end of buffer; leave it unterminated instead
      ;; of consuming its last character as a fake delimiter.
      (agl--propertize-backslashes-as-punctuation (1+ payload-start) region-end))
    (goto-char region-end)))

(defun agl--propertize-raw-interpolation-holes (start end)
  "Mark completed interpolation bodies in a raw-tail payload.

START and END delimit a payload already marked with
`agl-raw-tail-payload'.  Raw tails escape a `%{' whenever its immediately
preceding character is `\\', so this cannot reuse the template scanner."
  (save-excursion
    (goto-char start)
    (while (search-forward "%{" end t)
      (let ((open-start (match-beginning 0)) (body-start (match-end 0)))
        (unless (agl--escaped-interpolation-open-p open-start)
          (let ((depth 1) (p body-start))
            (while (and (> depth 0) (< p end))
              (cond
               ((eq (char-after p) ?\{) (setq depth (1+ depth)))
               ((eq (char-after p) ?\}) (setq depth (1- depth))))
              (setq p (1+ p)))
            (when (= depth 0)
              (put-text-property body-start (1- p) 'agl-interpolation-code t))))))))

(defun agl--propertize-backslashes-as-punctuation (start end)
  "Give every `\\' character between START and END punctuation syntax.

A raw-tail payload owns its backslashes as ordinary text rather than
treating them as AgL escape syntax (see
`docs/agl/reference/lexical-structure.md'), so a payload ending in `\\'
must not escape the fence character that closes it."
  (save-excursion
    (goto-char start)
    (while (search-forward "\\" end t)
      (put-text-property (1- (point)) (point) 'syntax-table (string-to-syntax ".")))))

(defun agl--propertize-raw-tail (name-start)
  "Propertize the raw-tail payload that follows the opener at NAME-START.

Point is right after a raw-tail opener (`exec$' or `ask$') that
started at NAME-START.  Skip a byte-adjacent `::[...]' type argument if
present, then propertize the payload -- the rest of the line, or a
following more-indented block -- as a generic string."
  (when (and (eq (char-after) ?:)
             (eq (char-after (1+ (point))) ?:)
             (eq (char-after (+ 2 (point))) ?\[))
    (forward-char 2)
    (condition-case nil
        (forward-list 1)
      (scan-error nil)))
  (let ((opener-indent (save-excursion (goto-char name-start) (current-indentation)))
        (opener-line-start (save-excursion (goto-char name-start) (line-beginning-position))))
    (skip-chars-forward " \t")
    (if (or (eolp) (eobp))
        (agl--propertize-raw-block opener-indent opener-line-start)
      (agl--propertize-raw-inline))))

(defun agl--skip-template-hole ()
  "Skip forward over a `%{...}' interpolation hole.

Point must be at the `%' of an unescaped `%{'.  Advance past the
matching close brace, counting nested braces, so the template scanner
does not mistake a quote or brace inside the hole -- e.g. the nested
call in `%{f(\"x\")}' -- for its own delimiter.  A hole may itself span
multiple lines.  If no matching close brace is found, stop at the end
of the buffer."
  (forward-char 2)                     ; past "%{"
  (let ((body-start (point)) (depth 1))
    (while (and (> depth 0) (not (eobp)))
      (cond
       ((eq (char-after) ?\{) (setq depth (1+ depth)) (forward-char 1))
       ((eq (char-after) ?\}) (setq depth (1- depth)) (forward-char 1))
       (t (forward-char 1))))
    (when (= depth 0)
      (put-text-property body-start (1- (point)) 'agl-interpolation-code t))))

(defun agl--template-hole-start-p ()
  "Return non-nil if point is at the `%' of an unescaped `%{' hole."
  (and (eq (char-after) ?%) (eq (char-after (1+ (point))) ?\{)))

(defun agl--propertize-single-quoted (quote-char)
  "Propertize the single-line template opened by QUOTE-CHAR at point.

If a matching, unescaped close quote is found before end of line, mark
both delimiters with string-quote syntax and move point past the
close quote.  Otherwise the template is unterminated: leave the
opener with its default (punctuation) syntax and advance one
character, so the template does not swallow the following lines.  A
`%{...}' hole is skipped whole (see `agl--skip-template-hole'), which
may carry the scan past the line-end limit if the hole spans multiple
lines."
  (let ((open (point))
        (limit (line-end-position))
        close)
    (forward-char 1)
    (while (and (not close) (< (point) limit))
      (cond
       ((eq (char-after) ?\\)
        (forward-char 1)
        (when (< (point) limit) (forward-char 1)))
       ((agl--template-hole-start-p) (agl--skip-template-hole))
       ((eq (char-after) quote-char) (setq close (point)))
       (t (forward-char 1))))
    (if close
        (progn
          (put-text-property open (1+ open) 'syntax-table (string-to-syntax "\""))
          (put-text-property close (1+ close) 'syntax-table (string-to-syntax "\""))
          (goto-char (1+ close)))
      (goto-char (1+ open)))))

(defun agl--propertize-triple-quoted (quote-char)
  "Propertize the triple-quoted template opened by QUOTE-CHAR at point.

Point is at the first of three QUOTE-CHAR characters.  Mark the first
character of the opening delimiter and the last character of the
closing delimiter with generic-string-fence syntax (python-mode's
technique for triple quotes), so inner single quotes and `#' are
inert.  If unterminated, fence the opener and treat the rest of the
buffer as inside the string, matching standard
unterminated-multiline-string handling.  A `%{...}' hole is skipped
whole (see `agl--skip-template-hole')."
  (let ((open (point)))
    (forward-char 3)
    (let (close)
      (while (and (not close) (not (eobp)))
        (cond
         ((eq (char-after) ?\\)
          (forward-char 1)
          (unless (eobp) (forward-char 1)))
         ((agl--template-hole-start-p) (agl--skip-template-hole))
         ((and (eq (char-after) quote-char)
               (eq (char-after (1+ (point))) quote-char)
               (eq (char-after (+ 2 (point))) quote-char))
          (forward-char 3)
          (setq close (point)))
         (t (forward-char 1))))
      (put-text-property open (1+ open) 'syntax-table (string-to-syntax "|"))
      (if close
          (progn
            (put-text-property (1- close) close 'syntax-table (string-to-syntax "|"))
            (when (> (line-number-at-pos close) (line-number-at-pos open))
              (put-text-property open close 'agl-multiline t)))
        (put-text-property open (point-max) 'agl-multiline t)))))

(defun agl--propertize-template ()
  "Propertize the template opened by the quote character at point.

Point is at a quote character in token-start position.  Dispatch to
the triple- or single-quoted handler and advance point past the
template."
  (let ((q (char-after)))
    (if (and (eq (char-after (1+ (point))) q)
             (eq (char-after (+ 2 (point))) q))
        (agl--propertize-triple-quoted q)
      (agl--propertize-single-quoted q))))

(defun agl-syntax-propertize-function (start end)
  "`syntax-propertize-function' for `agl-mode'.

Performs one left-to-right pass over START..END assigning
`syntax-table' text properties, consuming identifiers atomically so
that quotes and `#' inside them (e.g. `foo\"bar', `foo#bar') are never
visited as standalone characters.  See
`docs/agl/reference/lexical-structure.md' for the lexical rules this
implements.

A raw-tail opener (`exec$'/`ask$') is only recognized at bracket depth
zero, per the reference's \"only recognized at bracket depth zero\"
rule, so a bracket depth counter is threaded through the scan, seeded
from `(car (syntax-ppss start))' -- safe to call here because
`syntax-propertize' sets `syntax-propertize--done' to END before
invoking this function."
  (remove-text-properties start end '(agl-interpolation-code nil))
  (goto-char start)
  (let ((depth (car (syntax-ppss start)))
        case-fold-search)
    (while (< (point) end)
      (cond
       ((looking-at agl--ident-start-re)
        (let ((id-start (point)))
          (skip-chars-forward agl--ident-continue-skip)
          (when (and (= depth 0)
                     (member (buffer-substring-no-properties id-start (point))
                             agl-raw-tail-names))
            (agl--propertize-raw-tail id-start))))
       ((looking-at "[0-9]")
        (skip-chars-forward "0-9")
        (when (looking-at "\\.[0-9]")
          (forward-char 1)
          (skip-chars-forward "0-9")))
       ((eq (char-after) ?#)
        (put-text-property (point) (1+ (point)) 'syntax-table (string-to-syntax "<"))
        (goto-char (line-end-position)))
       ((memq (char-after) '(?\" ?\'))
        (agl--propertize-template))
       ((memq (char-after) '(?\( ?\[ ?\{))
        (setq depth (1+ depth))
        (forward-char 1))
       ((memq (char-after) '(?\) ?\] ?\}))
        (setq depth (max 0 (1- depth)))
        (forward-char 1))
       (t (forward-char 1))))))

(defun agl--propertize-extend-region (start end)
  "`syntax-propertize-extend-region-functions' entry for `agl-mode'.

If START falls inside a multi-line template or raw-tail block (marked
with the internal `agl-multiline' text property), move it back to the
beginning of that construct -- and in any case back to its line's
beginning -- so `agl-syntax-propertize-function' always resumes
scanning from a safe boundary, never from the middle of a multi-line
construct.  END is returned unchanged as the cdr of the result; this
function only ever adjusts the start of the region."
  (let ((new-start start))
    (when (and (> new-start (point-min))
               (get-text-property (1- new-start) 'agl-multiline))
      (setq new-start (or (previous-single-property-change
                            new-start 'agl-multiline nil (point-min))
                           (point-min))))
    (setq new-start (save-excursion (goto-char new-start) (line-beginning-position)))
    (if (< new-start start) (cons new-start end) nil)))

;; ---------------------------------------------------------------------------
;; Syntax table.
;; ---------------------------------------------------------------------------

(defvar agl-mode-syntax-table
  (let ((table (make-syntax-table)))
    ;; '#' and quotes are identifier-continuation characters (see
    ;; `agl--ident-continue-skip'), so comment/string syntax is assigned by
    ;; `agl-syntax-propertize-function', not by the table.
    (modify-syntax-entry ?# "." table)
    (modify-syntax-entry ?\n ">" table)
    (modify-syntax-entry ?\" "." table)
    (modify-syntax-entry ?\' "." table)
    ;; Identifier-continuation operator characters, so `forward-word' and
    ;; `thing-at-point' see whole AgL names such as `ask-prompt', `valid?',
    ;; and `a->b'.
    (modify-syntax-entry ?- "_" table)
    (modify-syntax-entry ?? "_" table)
    (modify-syntax-entry ?! "_" table)
    (modify-syntax-entry ?+ "_" table)
    (modify-syntax-entry ?* "_" table)
    (modify-syntax-entry ?< "_" table)
    (modify-syntax-entry ?> "_" table)
    (modify-syntax-entry ?_ "_" table)
    (modify-syntax-entry ?\\ "\\" table)
    (modify-syntax-entry ?\( "()" table)
    (modify-syntax-entry ?\) ")(" table)
    (modify-syntax-entry ?\[ "(]" table)
    (modify-syntax-entry ?\] ")[" table)
    (modify-syntax-entry ?\{ "(}" table)
    (modify-syntax-entry ?\} "){" table)
    table)
  "Syntax table for `agl-mode'.")

;; ---------------------------------------------------------------------------
;; Font-lock.
;;
;; Structural highlighting only : capitalization carries no syntactic
;; or semantic meaning in AgL (`option'/`Option' are equally valid as
;; types or values -- see docs/agl/reference/lexical-structure.md), so
;; faces derive only from declaration and annotation *positions*, never
;; from spelling.  Constructor use-sites inside expressions therefore stay
;; unfaced -- an accepted limit of lexical highlighting.  Every rule is a
;; function matcher built on `agl--search-ident-forward' rather than a
;; plain regexp, so matches respect AgL identifier boundaries instead of
;; `\\b'.  Font-lock's default OVERRIDE (nil) never replaces a face
;; already assigned by the syntactic (string/comment) pass, so these
;; rules leave template text and raw-tail payloads string-faced.  Completed
;; interpolation bodies are marked by the propertizer and receive the same
;; rules again with an override, restoring code faces only there.  The
;; interpolation-delimiter and attribute rules also deliberately override:
;; the former faces `%{' / `}', and the latter makes a whole `@name' read as
;; one attribute while rejecting string/comment candidates in its matcher.
;; ---------------------------------------------------------------------------

(defconst agl--keyword-face-names
  (append
   (let (result)
     (dolist (kw agl-keywords (nreverse result))
       (unless (member kw agl-constant-keywords)
         (push kw result))))
   agl-operator-keywords)
  "`agl-keywords' minus `agl-constant-keywords', plus `agl-operator-keywords'.

The words that get `font-lock-keyword-face' rather than
`font-lock-constant-face'.")

(defconst agl--keyword-face-re (regexp-opt agl--keyword-face-names)
  "Regexp matching one keyword-faced AgL word (excluding the literal constants).")

(defconst agl--constant-keyword-re (regexp-opt agl-constant-keywords)
  "Regexp matching one of `agl-constant-keywords'.")

(defconst agl--contextual-builtin-re (regexp-opt agl-contextual-builtins)
  "Regexp matching one of `agl-contextual-builtins'.")

(defconst agl--raw-tail-name-re (regexp-opt agl-raw-tail-names)
  "Regexp matching one of `agl-raw-tail-names'.")

(defconst agl--import-export-use-re (regexp-opt '("import" "export" "use"))
  "Regexp matching one of the import/export/use soft keywords.")

(defconst agl--type-annotation-anchor-re "\\(?::\\|->\\)"
  "Regexp matching the `:' or `->' that opens a type-annotation position.

Anchors `agl--match-type-annotation' to a parameter/field/return-type
annotation position (see docs/agl/reference/lexical-structure.md),
which is what makes that rule contextual rather than case-based .")

(defconst agl--number-re "[0-9]+\\(?:\\.[0-9]+\\)?"
  "Regexp matching an AgL `INT' or `DECIMAL' literal.")

(defconst agl--operator-name-excluded-chars "()[]{}:,.;\"'@#_"
  "The characters an AgL operator name may never contain.

Mirrors `_OPERATOR_NAME_EXCLUDED_CHARS' in
`src/agm/agl/lexer/scanner.py'.")

(defconst agl--colon-operator-re (regexp-opt '("::" ":="))
  "Regexp matching an AgL operator token that contains `:'.

`:' is excluded from operator names (`agl--operator-name-excluded-chars'),
so `::' and `:=' are fixed spellings rather than operator-name runs.  A
`:' always terminates an identifier, so neither can occur inside a name
and both are matched without an identifier-boundary check.")

(defconst agl--merged-operator-chars '(?/)
  "Operator characters the lexer merges into a longer token when unspaced.

`/' joins the segments of a module path (`std/text', `foo/bar::Point'),
which the lexer scans as one token rather than as an operator beside a
name.  A lone `/' is therefore faced only where an identifier does not
abut it; every other operator character is faced by position alone.

`@' is not here because it never reaches this test: it is one of
`agl--operator-name-excluded-chars', so `agl--operator-name-char-p'
rejects it and `@' is never faced as an operator at all.  It is faced
only as an attribute prefix (`agl--attribute-re').")

(defconst agl--attribute-re (concat "@" agl--name-re)
  "Regexp matching one declaration attribute's `@name' prefix.

An attribute is written `@name' or `@name(args)' in front of a defining
declaration, a parameter, or a field.  Only the prefix is faced: the
argument list is ordinary AgL and keeps its own faces.  Which names the
language defines is a static-analysis question, so any name faces here
-- and, because its rule faces with OVERRIDE, it does so even when an
earlier rule already claimed the name (`@copy', `@type').")

;; `font-lock-escape-face', `font-lock-number-face', and
;; `font-lock-operator-face' were all added in Emacs 29.1.
;; Package-Requires still floors at 27.1, so each is resolved through a
;; `facep' check with a pre-29 fallback rather than referenced directly
;; -- on 27/28 nothing errors either way, but referencing the absent face
;; directly would silently fontify with no face at all.

(defconst agl--escape-face
  (if (facep 'font-lock-escape-face) 'font-lock-escape-face 'font-lock-constant-face)
  "Face for `agl-interpolation-face' to inherit.

Falls back to `font-lock-constant-face', which has existed since
ancient Emacs, when `font-lock-escape-face' (added in 29.1) is
unavailable.")

(defconst agl--number-face
  (if (facep 'font-lock-number-face) 'font-lock-number-face 'font-lock-constant-face)
  "Face for AgL `INT'/`DECIMAL' literals.

Falls back to `font-lock-constant-face' -- the conventional pre-29
choice for numeric literals -- when `font-lock-number-face' (added in
29.1) is unavailable.")

(defconst agl--operator-face
  (if (facep 'font-lock-operator-face) 'font-lock-operator-face 'font-lock-builtin-face)
  "Face for AgL operator tokens.

Falls back to `font-lock-builtin-face' -- a conventional pre-29 choice
for operator highlighting -- when `font-lock-operator-face' (added in
29.1) is unavailable.")

(defface agl-interpolation-face
  `((t :inherit ,agl--escape-face))
  "Face for the `%{' and `}' delimiters of an AgL interpolation hole.

Hole contents (the expression between the delimiters) keep the
surrounding string/template face; fontifying AgL expressions inside a
hole is a documented non-goal (see
docs/agl/reference/lexical-structure.md).  Inherits `agl--escape-face'
-- a concrete, always-available face -- rather than
`font-lock-escape-face' directly, which does not exist before Emacs
29.1."
  :group 'agl)

(defun agl--item-start-p (pos)
  "Return non-nil if POS is the first non-whitespace column of its line."
  (save-excursion
    (goto-char pos)
    (skip-chars-backward " \t")
    (bolp)))

(defun agl--on-import-export-use-line-p (pos)
  "Return non-nil if POS's line begins, at item-start, with a soft keyword.

The soft keyword is `import', `export', or `use'.  This approximates
the `hiding' promotion window (\"within an import, use, or export
declaration\") as the physical line the keyword is written on."
  (save-excursion
    (goto-char pos)
    (beginning-of-line)
    (skip-chars-forward " \t")
    (and (looking-at agl--import-export-use-re)
         (agl--ident-boundary-after-p (match-end 0)))))

(defun agl--end-promoted-p (pos)
  "Return non-nil if the `end' match at POS is in its promotion window.

The window is item-start, followed by a `NAME (:: NAME)*' closer path."
  (and (agl--item-start-p pos)
       (save-excursion
         (goto-char (match-end 0))
         (skip-chars-forward " \t")
         (looking-at agl--name-re))))

(defun agl--match-reserved-keyword (limit)
  "`font-lock-keywords' MATCHER for keyword-faced AgL words, up to LIMIT."
  (agl--search-ident-forward agl--keyword-face-re limit))

(defun agl--match-constant-keyword (limit)
  "`font-lock-keywords' MATCHER for `true'/`false'/`null', up to LIMIT."
  (agl--search-ident-forward agl--constant-keyword-re limit))

(defun agl--match-contextual-builtin (limit)
  "`font-lock-keywords' MATCHER for `print'/`ask'/`exec', up to LIMIT."
  (agl--search-ident-forward agl--contextual-builtin-re limit))

(defun agl--match-raw-tail-name (limit)
  "`font-lock-keywords' MATCHER for `exec$'/`ask$', up to LIMIT."
  (agl--search-ident-forward agl--raw-tail-name-re limit))

(defun agl--match-use-keyword (limit)
  "`font-lock-keywords' MATCHER for item-start `use', up to LIMIT."
  (agl--search-ident-forward "use" limit #'agl--item-start-p))

(defun agl--match-export-keyword (limit)
  "`font-lock-keywords' MATCHER for item-start `export', up to LIMIT."
  (agl--search-ident-forward "export" limit #'agl--item-start-p))

(defun agl--match-scope-soft-keyword (limit)
  "`font-lock-keywords' MATCHER for item-start `scope', up to LIMIT."
  (agl--search-ident-forward "scope" limit #'agl--item-start-p))

(defun agl--match-import-keyword (limit)
  "`font-lock-keywords' MATCHER for item-start `import', up to LIMIT."
  (agl--search-ident-forward "import" limit #'agl--item-start-p))

(defun agl--match-hiding-keyword (limit)
  "`font-lock-keywords' MATCHER for promoted `hiding', up to LIMIT."
  (agl--search-ident-forward "hiding" limit #'agl--on-import-export-use-line-p))

(defun agl--match-end-keyword (limit)
  "`font-lock-keywords' MATCHER for promoted `end', up to LIMIT."
  (agl--search-ident-forward "end" limit #'agl--end-promoted-p))

(defun agl--parse-type-head-chain ()
  "Parse a type expression's qualifier chain at point, consuming it.

The shape is `qualifier_chain? name', where `qualifier_chain' is one or
more `[\"/\"] NAME (\"/\" NAME)* \"::\"' segments (see
docs/agl/reference/grammar.md's `qualifier_chain' and `type_expr').
This is broader than a `decl_head' (see `agl--parse-decl-head-chain',
used for declaration positions): a type expression may also route
through a `/'-separated module path, as in `foo/bar::Point', which a
`decl_head' never does.  Point moves to the end of the chain.  Return a
list (FULL-START SEG-START SEG-END), where FULL-START begins the first
segment and SEG-START/SEG-END bound only the terminal segment -- the
name actually faced -- or nil if point is not at a NAME."
  (skip-chars-forward " \t\n")
  (when (looking-at agl--ident-start-re)
    (let ((full-start (point)) seg-start seg-end)
      (setq seg-start (point))
      (skip-chars-forward agl--ident-continue-skip)
      (setq seg-end (point))
      (while (cond
              ((looking-at "::[[:alpha:]_]") (forward-char 2) t)
              ((looking-at "/[[:alpha:]_]") (forward-char 1) t))
        (setq seg-start (point))
        (skip-chars-forward agl--ident-continue-skip)
        (setq seg-end (point)))
      (list full-start seg-start seg-end))))

(defun agl--qualifier-colon-p (pos anchor-end)
  "Return non-nil when the `:' at POS belongs to a `::' qualifier.

ANCHOR-END is the end of the matched anchor.  `::' separates a qualifier
chain from the member it selects, so the name after it is an ordinary
reference rather than a type annotation.  Either colon of the pair is
rejected: the scan resumes inside the pair after the first one fails."
  (or (eq (char-after anchor-end) ?:)
      (eq (char-before pos) ?:)))

(defun agl--dict-entry-colon-p (pos)
  "Return non-nil if the `:' at POS separates a dict-literal entry.

A `:' whose innermost enclosing bracket is a brace belongs to a dict
literal (`{ key: value }'), where the value is an ordinary expression
rather than a type annotation.  Parameter and field lists use
parentheses, and layout-form field blocks and return types are not
inside a brace at all, so this rejects only the dict case."
  (save-excursion
    (let ((open (nth 1 (syntax-ppss pos))))
      (and open (eq (char-after open) ?\{)))))

(defun agl--match-type-annotation (limit)
  "`font-lock-keywords' MATCHER for a type head in annotation position.

Searches up to LIMIT.

Matches the head name of the type expression after `:' (in a parameter
or field list) or after `->' (a return-type annotation): a `NAME' with
an optional qualifier chain (`::' segments, and optionally a `/'
module route -- see `agl--parse-type-head-chain') and optional
`[...]' type arguments.  Only the terminal segment is faced, matching
how the declared-name matchers treat a qualifier prefix (position,
not spelling, drives the face) -- so in `foo/bar::Point' only `Point'
is faced.  The eight primitive type-annotation names
\(`agl-primitive-type-names') are ordinary `NAME's, so this single rule
covers them too; there is no separate primitive-only case.  `match-data'
group 1 covers the terminal segment.  Return non-nil on success."
  (let (found)
    (while (and (not found) (re-search-forward agl--type-annotation-anchor-re limit t))
      (let ((anchor-end (match-end 0)))
        (goto-char anchor-end)
        (if (or (agl--qualifier-colon-p (match-beginning 0) anchor-end)
                (agl--dict-entry-colon-p (match-beginning 0)))
            ;; A `:' directly inside a brace is a dict entry (`{ key: value }'),
            ;; not an annotation, so its value is an ordinary expression.
            nil
        (skip-chars-forward " \t\n")
        (let ((chain (and (looking-at agl--ident-start-re)
                           (agl--parse-type-head-chain))))
          (if (and chain (agl--ident-boundary-before-p (nth 0 chain)))
              (let ((seg-start (nth 1 chain)) (seg-end (nth 2 chain)))
                (set-match-data (list seg-start seg-end seg-start seg-end))
                (goto-char seg-end)
                (setq found t))
            (goto-char anchor-end))))))
    found))

(defun agl--match-number (limit)
  "`font-lock-keywords' MATCHER for an `INT'/`DECIMAL' literal, up to LIMIT."
  (agl--search-ident-forward agl--number-re limit))

(defun agl--operator-name-char-p (char)
  "Return non-nil when CHAR may take part in an AgL operator name.

Mirrors `_is_operator_name_char' in `src/agm/agl/lexer/scanner.py': a
Unicode punctuation or symbol character that is not one of
`agl--operator-name-excluded-chars'.  Whitespace is neither punctuation
nor a symbol, so it is excluded by the category test itself."
  (and char
       (not (memq char (append agl--operator-name-excluded-chars nil)))
       (memq (aref (symbol-name (get-char-code-property char 'general-category)) 0)
             '(?P ?S))))

(defun agl--operator-token-start-p (pos)
  "Return non-nil when POS begins a token rather than continuing a name.

An identifier consumes every character outside `IDENT_STOP' (see
`agl--ident-continue-skip'), so an operator character in the middle of
such a run belongs to the name: `a+b' is one identifier, not `a', `+',
`b'.  POS begins a token when its own character terminates an identifier
\(`=', `|' and `/' are operator characters that are also `IDENT_STOP'
members), when the run POS sits in starts at POS, or when that run began
with a digit and its number token ends exactly at POS -- the `+' of
`1+2' follows a complete `INT'."
  (or (memq (char-after pos) agl--ident-stop-char-list)
      (save-excursion
        (goto-char pos)
        (skip-chars-backward agl--ident-continue-skip)
        (or (= (point) pos)
            (and (looking-at-p "[0-9]")
                 (progn (skip-chars-forward "0-9") (= (point) pos)))))))

(defun agl--operator-run-end (start limit)
  "Return the end of the operator-name run beginning at START, before LIMIT.

The run is maximal, which is what makes a multi-character operator name
-- `|>', `<|', `>>', or any spelling a program declares with `infixl' /
`infixr' -- one match rather than one match per character.  A lone `?'
followed by ASCII digits is extended over them, mirroring the scanner's
`PLACEHOLDER_NUM': `?1' is a single placeholder token, not `?' beside
the number 1."
  (let ((end start))
    (while (and (< end limit) (agl--operator-name-char-p (char-after end)))
      (setq end (1+ end)))
    (when (and (= end (1+ start)) (eq (char-after start) ??))
      (save-excursion
        (goto-char end)
        (skip-chars-forward "0-9" limit)
        (setq end (point))))
    end))

(defun agl--match-operator (limit)
  "`font-lock-keywords' MATCHER for one AgL operator token, up to LIMIT.

Two shapes are recognized.  `::' and `:=' contain `:', which no operator
name may (`agl--colon-operator-re'), so they are matched as fixed
spellings.  Every other operator token is an operator name: a maximal
run of operator characters (`agl--operator-run-end') that begins a token
\(`agl--operator-token-start-p').  Scanning the whole run is what keeps a
multi-character spelling one match, and the token-start test is what
keeps the `+' of the single identifier `a+b' unfaced.

A run that is only `/' is faced solely when no identifier abuts it:
unspaced, the lexer merges it into a module path rather than emitting an
operator -- see `agl--merged-operator-chars'."
  (let ((found nil))
    (while (and (not found) (< (point) limit))
      (let ((pos (point)))
        (cond
         ((and (looking-at agl--colon-operator-re) (<= (match-end 0) limit))
          (goto-char (match-end 0))
          (set-match-data (list pos (point)))
          (setq found t))
         ((and (agl--operator-name-char-p (char-after pos))
               (agl--operator-token-start-p pos))
          (let ((end (agl--operator-run-end pos limit)))
            (if (and (= end (1+ pos))
                     (memq (char-after pos) agl--merged-operator-chars)
                     (not (and (agl--ident-boundary-before-p pos)
                               (agl--ident-boundary-after-p end))))
                (goto-char end)
              (goto-char end)
              (set-match-data (list pos end))
              (setq found t))))
         (t
          (forward-char 1)
          (skip-chars-forward "[:alnum:] \t\n_" limit)))))
    found))

(defun agl--match-attribute (limit)
  "`font-lock-keywords' MATCHER for an attribute's `@name', up to LIMIT.

The rule this drives faces with OVERRIDE so that the whole `@name' takes
the attribute face even where an earlier rule already faced the name
\(`@copy' as a builtin, `@type' as a keyword).  An override also outranks
the syntactic pass, which must not happen, so a candidate inside a
string, template, raw-tail payload or comment is rejected here instead.
The test is one character past the `@' for the reason spelled out in
`agl--decl-head-candidate-rejected-p'; `@name' is always at least two
characters, so that position is still inside the same region."
  (agl--search-ident-forward
   agl--attribute-re limit
   (lambda (start) (not (agl--in-string-or-comment-p (1+ start))))))

(defun agl--in-string-p (pos)
  "Return non-nil if POS is inside a string/template per `syntax-ppss'."
  (nth 3 (syntax-ppss pos)))

(defun agl--in-string-or-comment-p (pos)
  "Return non-nil if POS is inside a string/template or comment.

Per `syntax-ppss': `nth 3' flags a string/template (including a
raw-tail payload, which carries generic-string-fence syntax), `nth 4' a
comment.  Used to reject a decl-head candidate that is only text -- a
commented-out declaration, or one embedded in a template or raw-tail
payload -- from `agl--match-attribute', `agl-imenu-create-index' and
`agl--toplevel-line-p'."
  (let ((state (syntax-ppss pos)))
    (or (nth 3 state) (nth 4 state))))

(defun agl--string-region-end (pos)
  "Return the end of the string/template region enclosing POS.

POS must satisfy `agl--in-string-p'.  The region's end is found with
`forward-sexp' from the region's syntax-recorded start
\(`(nth 8 (syntax-ppss POS))'), which handles both quote-character
strings and generic-string-fence regions (triple-quoted templates,
raw-tail payloads) uniformly, since both kinds are balanced sexps under
`parse-sexp-lookup-properties' (non-nil by default, which is what makes
the `syntax-table' text properties this file assigns visible to the
sexp scanner at all).  An unterminated region has no matching close, so
`forward-sexp' signals `scan-error'; this returns `point-max' in that
case, matching how the propertize layer treats the rest of the buffer
as inside an unterminated region."
  (let ((region-start (nth 8 (syntax-ppss pos))))
    (save-excursion
      (goto-char region-start)
      (condition-case nil
          (progn (forward-sexp 1) (point))
        (scan-error (point-max))))))

(defun agl--preceding-backslash-parity-odd-p (pos)
  "Return non-nil if an odd run of `\\' characters precedes POS.

An odd run means POS itself is escaped.  This is *template* escape
semantics (docs/agl/reference/strings-and-interpolation.md): `\\\\' is
an escaped backslash, so backslashes pair off and only a leftover,
unpaired one escapes what follows.  See
`agl--escaped-interpolation-open-p' for the raw-tail-payload semantics,
which are different."
  (let ((count 0) (p pos))
    (while (and (> p (point-min)) (eq (char-before p) ?\\))
      (setq count (1+ count) p (1- p)))
    (= 1 (mod count 2))))

(defun agl--escaped-interpolation-open-p (pos)
  "Return non-nil if the `%{' hole opener at POS is escaped.

Dispatches on whether POS lies in a raw-tail payload (tagged with the
internal `agl-raw-tail-payload' text property -- see
`agl--propertize-raw-inline'/`agl--propertize-raw-block') or a template
\(single-line or triple-quoted), since the two have different backslash
semantics.  A template pairs backslashes
\(`agl--preceding-backslash-parity-odd-p'): `\\\\%{' is an unescaped
hole.  A raw-tail payload instead \"owns\" its
ordinary backslashes (docs/agl/reference/lexical-structure.md's
raw-tail-forms section), matching the real scanner
\(`src/agm/agl/lexer/scanner.py'): any single immediately preceding `\\'
escapes the hole, with no parity counting, so `\\\\%{' is still escaped
there."
  (if (get-text-property pos 'agl-raw-tail-payload)
      (and (> pos (point-min)) (eq (char-before pos) ?\\))
    (agl--preceding-backslash-parity-odd-p pos)))

(defun agl--match-interpolation-delims (limit)
  "Search forward for a `%{...}' interpolation hole, up to LIMIT.

Only matches inside a string/template region.  `match-data' group 1
covers the opening `%{' and group 2 covers the closing `}'; hole
contents are left with their inherited string face (see
`agl-interpolation-face').  An escaped `\\%{'
\(docs/agl/reference/lexical-structure.md, and see
`agl--escaped-interpolation-open-p' for the raw-tail-payload vs.
template distinction) is skipped.  The closing-brace scan is bounded by
the enclosing string region's own end (`agl--string-region-end'), not
just LIMIT or `point-max': an unbalanced `%{' -- the normal transient
state while typing one -- must never scan, or paint
`agl-interpolation-face' (its rule uses OVERRIDE=t), past the string it
opened in.  Return non-nil on success."
  (let (found)
    (while (and (not found) (search-forward "%{" limit t))
      (let ((open-start (match-beginning 0)) (open-end (match-end 0)))
        (if (and (agl--in-string-p open-start)
                 (not (agl--escaped-interpolation-open-p open-start)))
            (let ((depth 1) (p open-end)
                  (region-end (min limit (agl--string-region-end open-start)))
                  close-start close-end)
              (while (and (> depth 0) (< p region-end))
                (cond
                 ((eq (char-after p) ?\{) (setq depth (1+ depth)) (setq p (1+ p)))
                 ((eq (char-after p) ?\}) (setq depth (1- depth)) (setq p (1+ p)))
                 (t (setq p (1+ p)))))
              (if (= depth 0)
                  (progn
                    (setq close-end p close-start (1- p))
                    (set-match-data (list open-start close-end
                                          open-start open-end
                                          close-start close-end))
                    (goto-char close-end)
                    (setq found t))
                (goto-char open-end)))
          (goto-char open-end))))
    found))

(defun agl--decl-head-segment-start-p ()
  "Return non-nil when point is at the start of a `decl_head' segment.

A `name' is a `NAME' or an `OP_NAME' (see docs/agl/reference/grammar.md's
`name'), so a segment begins at an identifier character or at an operator
character -- the stdlib declares `def |>[A, B](...)' and its siblings that
way."
  (or (looking-at-p agl--ident-start-re)
      (agl--operator-name-char-p (char-after))))

(defun agl--decl-head-segment-end ()
  "Consume the `decl_head' segment at point and return its end position.

An identifier runs to the next `IDENT_STOP' character; an operator name
runs to the end of its operator-character run (`agl--operator-run-end'),
which is what stops `def |>[A, B]' at the type-argument bracket."
  (if (looking-at-p agl--ident-start-re)
      (skip-chars-forward agl--ident-continue-skip)
    (goto-char (agl--operator-run-end (point) (point-max))))
  (point))

(defun agl--parse-decl-head-chain ()
  "Parse a `decl_head'-shaped qualifier chain at point, consuming it.

The shape is `[scope_path \"::\"] name' (see
docs/agl/reference/grammar.md's `decl_head'); point moves to its end.
Return a list (FULL-START SEG-START SEG-END), where FULL-START begins
the first segment and SEG-START/SEG-END bound only the terminal
segment, or nil if point is not at a `name'."
  (skip-chars-forward " \t\n")
  (when (agl--decl-head-segment-start-p)
    (let ((full-start (point)) seg-start seg-end)
      (setq seg-start (point))
      (setq seg-end (agl--decl-head-segment-end))
      (while (and (looking-at-p "::")
                  (save-excursion (forward-char 2) (agl--decl-head-segment-start-p)))
        (forward-char 2)
        (setq seg-start (point))
        (setq seg-end (agl--decl-head-segment-end)))
      (list full-start seg-start seg-end))))

(defun agl--decl-head-pattern-p (chain-end)
  "Return non-nil if a decl-head chain ending at CHAIN-END names a pattern.

Per docs/agl/reference/scopes.md's binder-path table (\"Writing an
argument list, even an empty one, an `as' binder, ... keeps the
pattern's ordinary match meaning\"), `let'/`var' followed by a
qualifier chain and then `(' or `as' names a constructor pattern, not a
scoped/root binding -- `let Point(x, y) = p' matches a `Point' pattern,
it does not bind a variable named `Point'.  Checked, for symmetry, the
same way `agl--search-catch-binder' checks for a trailing `as'."
  (save-excursion
    (goto-char chain-end)
    (or (eq (char-after) ?\()
        (progn
          (skip-chars-forward " \t\n")
          (and (looking-at "as") (agl--ident-boundary-after-p (match-end 0)))))))

(defun agl--search-decl-head (keyword limit &optional require-item-start reject-pattern)
  "Search forward for boundary-safe KEYWORD followed by a decl-head chain.

Search is bounded by LIMIT.  Used for `def'/`record'/`enum'/`type'/
`exception'/`let'/`var' (font-lock declared-name faces) and for
`scope'/`end' (imenu's scope-nesting tracker) -- the single matcher
both font-lock and imenu are built on.

Sets `match-data' with four groups: 0 spans the whole match (keyword
through the terminal name), 1 is the keyword itself, 2 is the full
qualifier chain as written (e.g. `Box::get'), and 3 is only its
terminal segment (e.g. `get') -- the position font-lock faces as the
declared name.  When REQUIRE-ITEM-START is non-nil, KEYWORD must also be
the first token on its line (see `agl--item-start-p').  When
REJECT-PATTERN is non-nil (`let'/`var' only -- see
`agl--decl-head-pattern-p'), a chain that names a constructor pattern
rather than a binding is not accepted.  Return non-nil on success, with
point left at the end of group 3."
  (let ((kw-re (regexp-quote keyword)) found)
    (while (and (not found) (agl--search-ident-forward kw-re limit))
      (let ((kw-start (match-beginning 0)) (kw-end (match-end 0)))
        (when (or (not require-item-start) (agl--item-start-p kw-start))
          (let ((chain (agl--parse-decl-head-chain)))
            (when (and chain
                       (not (and reject-pattern (agl--decl-head-pattern-p (nth 2 chain)))))
              (let ((full-start (nth 0 chain)) (seg-start (nth 1 chain)) (seg-end (nth 2 chain)))
                (set-match-data (list kw-start seg-end
                                       kw-start kw-end
                                       full-start seg-end
                                       seg-start seg-end))
                (goto-char seg-end)
                (setq found t)))))))
    found))

(defun agl--search-catch-binder (limit)
  "Search forward for a `catch' clause's `as' alias, up to LIMIT.

Only the alias introduced by `as' is a true binder: `catch_pattern' is
`name (\"as\" name)?', and per docs/agl/reference/exceptions.md, \"`as
name' binds the exception as `name'\" -- the leading name/`_' matches an
exception type, it does not bind one.  So `catch NotFound => ...' has
nothing to fontify, while `catch NotFound as e => ...' faces only `e'.
Sets `match-data' so group 1 covers the alias.  Return non-nil on
success."
  (let (found)
    (while (and (not found) (agl--search-ident-forward "catch" limit))
      (skip-chars-forward " \t\n")
      (when (looking-at agl--name-re)
        (goto-char (match-end 0))
        (skip-chars-forward " \t\n")
        (when (and (looking-at "as") (agl--ident-boundary-after-p (match-end 0)))
          (goto-char (match-end 0))
          (skip-chars-forward " \t\n")
          (when (looking-at agl--name-re)
            (set-match-data (list (match-beginning 0) (match-end 0)
                                   (match-beginning 0) (match-end 0)))
            (goto-char (match-end 0))
            (setq found t)))))
    found))

(defconst agl--code-font-lock-rules
  (list
   (list #'agl--match-reserved-keyword 'font-lock-keyword-face)
   (list #'agl--match-constant-keyword 'font-lock-constant-face)
   (list #'agl--match-import-keyword 'font-lock-keyword-face)
   (list #'agl--match-use-keyword 'font-lock-keyword-face)
   (list #'agl--match-export-keyword 'font-lock-keyword-face)
   (list #'agl--match-hiding-keyword 'font-lock-keyword-face)
   (list #'agl--match-scope-soft-keyword 'font-lock-keyword-face)
   (list #'agl--match-end-keyword 'font-lock-keyword-face)
   (list #'agl--match-contextual-builtin 'font-lock-builtin-face)
   (list #'agl--match-raw-tail-name 'font-lock-builtin-face)
   (list (lambda (limit) (agl--search-decl-head "def" limit)) 'font-lock-function-name-face 3)
   (list (lambda (limit) (agl--search-decl-head "record" limit)) 'font-lock-type-face 3)
   (list (lambda (limit) (agl--search-decl-head "enum" limit)) 'font-lock-type-face 3)
   (list (lambda (limit) (agl--search-decl-head "type" limit)) 'font-lock-type-face 3)
   (list (lambda (limit) (agl--search-decl-head "exception" limit)) 'font-lock-type-face 3)
   (list (lambda (limit) (agl--search-decl-head "let" limit nil t))
         'font-lock-variable-name-face 3)
   (list (lambda (limit) (agl--search-decl-head "var" limit nil t))
         'font-lock-variable-name-face 3)
   (list #'agl--search-catch-binder 'font-lock-variable-name-face 1)
   (list #'agl--match-type-annotation 'font-lock-type-face 1)
   ;; The operator rule precedes the number rule so that a `?N' placeholder
   ;; faces as the one token it is; otherwise its digits would already carry
   ;; the number face, which font-lock's default OVERRIDE never replaces.
   (list #'agl--match-operator 'agl--operator-face)
   (list #'agl--match-number 'agl--number-face)
   (list #'agl--match-attribute 'font-lock-preprocessor-face 0 t))
  "Font-lock rules shared by ordinary code and interpolation bodies.")

(defun agl--match-interpolation-code (matcher limit)
  "Run MATCHER until it finds a match inside an interpolation body, up to LIMIT."
  (let (found)
    (while (and (not found) (funcall matcher limit))
      (when (get-text-property (match-beginning 0) 'agl-interpolation-code)
        (setq found t)))
    found))

(defun agl--match-interpolation-body (limit)
  "Match one contiguous interpolation body, up to LIMIT.

The syntactic pass gives an enclosing template a string face.  This matcher
clears that face before the ordinary code rules add their more specific faces."
  (let (start end)
    (while (and (not start) (< (point) limit))
      (if (get-text-property (point) 'agl-interpolation-code)
          (setq start (point)
                end (min limit
                         (next-single-property-change
                          (point) 'agl-interpolation-code nil (point-max))))
        (goto-char (min limit
                         (next-single-property-change
                          (point) 'agl-interpolation-code nil (point-max))))))
    (when start
      (set-match-data (list start end))
      (goto-char end)
      t)))

(defun agl--font-lock-rule (rule &optional interpolation-only)
  "Build a font-lock rule from RULE, optionally restricted to interpolation code."
  (let* ((matcher (nth 0 rule))
         (face (nth 1 rule))
         (subexp (or (nth 2 rule) 0))
         (override (or interpolation-only (nth 3 rule)))
         (effective-matcher
          (if interpolation-only
              (lambda (limit) (agl--match-interpolation-code matcher limit))
            matcher)))
    (list effective-matcher
          (append (list subexp face) (when override '(t))))))

(defconst agl-font-lock-keywords
  (append
   (mapcar #'agl--font-lock-rule agl--code-font-lock-rules)
   (list (list #'agl--match-interpolation-delims
               '(1 'agl-interpolation-face t)
               '(2 'agl-interpolation-face t))
         (list #'agl--match-interpolation-body '(0 nil t)))
   (mapcar (lambda (rule) (agl--font-lock-rule rule t)) agl--code-font-lock-rules))
  "Font-lock keyword rules for `agl-mode'.

See the section commentary above this constant for the governing
design (structural highlighting only).")

;; ---------------------------------------------------------------------------
;; imenu and defun navigation.
;; ---------------------------------------------------------------------------

(defun agl--decl-head-preceded-by-program-p (kw-start)
  "Return non-nil if `program' precedes KW-START, modulo whitespace.

KW-START is a `def' match's keyword start; `program def' is the
Programs imenu category, while a bare `def'/`extern def' is Functions."
  (save-excursion
    (goto-char kw-start)
    (skip-chars-backward " \t\n")
    (let ((end (point)))
      (and (>= (- end 7) (point-min))
           (string= "program" (buffer-substring-no-properties (- end 7) end))
           (agl--ident-boundary-before-p (- end 7))))))

(defun agl--decl-head-candidate-rejected-p (kw-start)
  "Return non-nil if a decl-head candidate starting at KW-START is only text.

Checked one character past KW-START, not at KW-START itself:
`syntax-ppss' reports the state as of just *before* a position, so the
state exactly at a region's first character would read as \"not
inside\" even when that character is itself the region's content --
concretely, the first character of an inline raw-tail payload doubles
as that payload's synthetic opening fence (see
`agl--propertize-raw-inline'), so a decl-head keyword landing exactly
there (`exec$ def fake()') would otherwise slip through unrejected.
Checking one character in is always still inside the same region
for any keyword this file matches against a decl head (they are all
longer than one character), and is never inside a *different* region
for an ordinary, non-embedded declaration.  Uses `save-match-data':
`agl--in-string-or-comment-p' calls `syntax-ppss', which the caller
cannot assume leaves `match-data' alone."
  (save-match-data (agl--in-string-or-comment-p (1+ kw-start))))

(defun agl--qualify-decl-name (name scope-stack)
  "Prefix NAME with SCOPE-STACK's accumulated path, outer scope first.

SCOPE-STACK holds innermost-first (its `car' is the innermost currently
open `scope' region's own path text, per `agl--search-decl-head')."
  (if scope-stack
      (concat (mapconcat #'identity (reverse scope-stack) "::") "::" name)
    name))

(defun agl-imenu-create-index ()
  "`imenu-create-index-function' for `agl-mode'.

Categories: Programs (`program def'), Functions (`def', `extern def',
`builtin def'), Types (`record'/`enum'/`type'/`exception'), and Scopes
\(`scope' paths).  A declaration inside an open `scope' region, or
written with its own `Type::member' qualifier, indexes under a
qualified name (e.g. `Point::distance').  Reuses `agl--search-decl-head'
-- the same declaration matcher `agl-font-lock-keywords' calls -- driven
line-by-line here so open/close `scope' regions can be tracked as a
stack in text order."
  (let (programs functions types scopes scope-stack)
    (save-excursion
      (goto-char (point-min))
      (while (not (eobp))
        (let ((line-start (point)) (line-end (line-end-position)))
          ;; Each `agl--search-decl-head' call below must start from
          ;; LINE-START: when the previous attempt's match is rejected
          ;; (e.g. the "end" inside "extends"), `agl--search-ident-forward'
          ;; leaves point wherever its last rejected retry landed, not
          ;; back at its original position -- `re-search-forward' itself
          ;; only guarantees that on an *outright* failed call, and here
          ;; the calls that matter are the ones after an internal retry.
          (cond
           ((and (progn (goto-char line-start) (agl--search-decl-head "scope" line-end t))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (let* ((chain (match-string-no-properties 2))
                   (qualified (agl--qualify-decl-name chain scope-stack)))
              (push (cons qualified (copy-marker (match-beginning 2))) scopes)
              (push chain scope-stack)))
           ((and (progn (goto-char line-start) (agl--search-decl-head "end" line-end t))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (let ((chain (match-string-no-properties 2)))
              (when (and scope-stack (string= (car scope-stack) chain))
                (pop scope-stack))))
           ((and (progn (goto-char line-start) (agl--search-decl-head "def" line-end))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (let* ((chain (match-string-no-properties 2))
                   (qualified (agl--qualify-decl-name chain scope-stack))
                   (marker (copy-marker (match-beginning 3)))
                   (entry (cons qualified marker)))
              (if (agl--decl-head-preceded-by-program-p (match-beginning 1))
                  (push entry programs)
                (push entry functions))))
           ((and (progn (goto-char line-start) (agl--search-decl-head "record" line-end))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (push (cons (agl--qualify-decl-name (match-string-no-properties 2) scope-stack)
                        (copy-marker (match-beginning 3)))
                  types))
           ((and (progn (goto-char line-start) (agl--search-decl-head "enum" line-end))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (push (cons (agl--qualify-decl-name (match-string-no-properties 2) scope-stack)
                        (copy-marker (match-beginning 3)))
                  types))
           ((and (progn (goto-char line-start) (agl--search-decl-head "type" line-end))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (push (cons (agl--qualify-decl-name (match-string-no-properties 2) scope-stack)
                        (copy-marker (match-beginning 3)))
                  types))
           ((and (progn (goto-char line-start) (agl--search-decl-head "exception" line-end))
                 (not (agl--decl-head-candidate-rejected-p (match-beginning 0))))
            (push (cons (agl--qualify-decl-name (match-string-no-properties 2) scope-stack)
                        (copy-marker (match-beginning 3)))
                  types)))
          (goto-char line-start)
          (forward-line 1))))
    (delq nil
          (list (when programs (cons "Programs" (nreverse programs)))
                (when functions (cons "Functions" (nreverse functions)))
                (when types (cons "Types" (nreverse types)))
                (when scopes (cons "Scopes" (nreverse scopes)))))))

(defun agl--toplevel-line-p ()
  "Return non-nil if point's line is a top-level AgL declaration line.

Such a line begins at column 0 with a non-blank character that starts
neither a comment nor a string/template/raw-tail-payload region.
Checked via `agl--in-string-or-comment-p' at the position right after
that character -- not at its own position, since `syntax-ppss' reports
the state *before* a character is consumed, so a comment-opening `#'
itself only registers as \"inside a comment\" one position later -- so a
literal `#' test is unnecessary here; a comment-only line is correctly
rejected by the general check."
  (save-excursion
    (beginning-of-line)
    (and (looking-at "[^ \t\n]")
         (not (agl--in-string-or-comment-p (match-end 0))))))

(defun agl--search-toplevel-line-backward ()
  "Move point to the beginning of the nearest earlier top-level line.

Skips a candidate beginning-of-line that turns out to be inside a
string/template, raw-tail payload, or comment (see
`agl--toplevel-line-p') -- e.g. a continuation line of a multi-line
string that merely looks top-level.  Return non-nil on success, leaving
point unmoved on failure."
  (let ((start (point)) found)
    (while (and (not found) (re-search-backward "^[^ \t\n]" nil t))
      (if (agl--toplevel-line-p)
          (setq found t)
        (goto-char (1- (point)))))
    (unless found (goto-char start))
    found))

(defun agl--goto-defun-start-backward ()
  "Move point to the start of the nearest top-level AgL declaration line.

The target line is at or before point, skipping the line point is on
when point is already at that line's very beginning.  Return non-nil
on success."
  (let ((bol (line-beginning-position)))
    (when (and (= (point) bol) (not (bobp)))
      (goto-char (1- (point))))
    (beginning-of-line)
    (if (agl--toplevel-line-p)
        t
      (agl--search-toplevel-line-backward))))

(defun agl--goto-defun-end-forward ()
  "Move point to just before the next top-level AgL declaration line.

Move to `point-max' instead if there is no such line."
  (let (found)
    (while (and (not found) (not (eobp)))
      (forward-line 1)
      (when (or (eobp) (agl--toplevel-line-p)) (setq found t)))
    (unless found (goto-char (point-max)))
    t))

(defun agl-beginning-of-defun (&optional arg)
  "`beginning-of-defun-function' for `agl-mode'.

Move point ARG (default 1) top-level AgL declarations backward, or
forward if ARG is negative -- the contract `beginning-of-defun-function'
documents (it is called with the same ARG `beginning-of-defun' itself
receives, unlike `end-of-defun-function', which Emacs always calls with
no argument -- see `agl-end-of-defun').  A declaration starts at column
0 (AgL is layout-sensitive, so a declaration ends where a line returns
to its own indentation or less).  Return non-nil if point moved; nil if
it did not (e.g. ARG is 1 and point was already at the start of a
top-level declaration, such as `point-min') -- the safer contract for a
caller that loops on the return value."
  (setq arg (or arg 1))
  (let ((start (point)))
    (if (< arg 0)
        (dotimes (_ (- arg)) (agl--goto-defun-end-forward))
      (dotimes (_ arg) (agl--goto-defun-start-backward)))
    (/= (point) start)))

(defun agl-end-of-defun ()
  "`end-of-defun-function' for `agl-mode'.

Move point to just before the next top-level AgL declaration line (or
to `point-max' if there is none).  Takes no argument: Emacs always
calls `end-of-defun-function' with none -- `end-of-defun' itself handles
any ARG and negative-ARG looping by calling `beginning-of-defun-raw'
\(which uses `agl-beginning-of-defun') before each call to this
function."
  (agl--goto-defun-end-forward))

;; ---------------------------------------------------------------------------
;; Major mode definition.
;; ---------------------------------------------------------------------------

;;;###autoload (add-to-list 'auto-mode-alist '("\\.agl\\'" . agl-mode))

;;;###autoload
(define-derived-mode agl-mode prog-mode "AgL"
  "Major mode for editing AgL source files."
  :syntax-table agl-mode-syntax-table
  (setq-local comment-start "# ")
  (setq-local comment-start-skip "#+[ \t]*")
  (setq-local comment-end "")
  (setq-local indent-tabs-mode nil)
  ;; AgL layout counts a tab as advancing to the next multiple of 4 columns.
  (setq-local tab-width 4)
  (setq-local syntax-propertize-function #'agl-syntax-propertize-function)
  (add-hook 'syntax-propertize-extend-region-functions
            #'agl--propertize-extend-region nil t)
  (setq-local font-lock-defaults '(agl-font-lock-keywords nil nil))
  (setq-local imenu-create-index-function #'agl-imenu-create-index)
  (setq-local beginning-of-defun-function #'agl-beginning-of-defun)
  (setq-local end-of-defun-function #'agl-end-of-defun)
  (setq-local indent-line-function #'agl-indent-line-function)
  (setq-local indent-region-function #'agl-indent-region)
  ;; RET indents the line it opens, but must not re-indent the line it ends.
  ;; A layout language gives a line several legal columns, and the one the
  ;; engine computes is only the likeliest: rewriting the finished line to it
  ;; would undo the dedent that starts a new declaration and flatten a body
  ;; deliberately indented wider than `agl-indent-offset'.
  (setq-local electric-indent-inhibit t)
  (add-hook 'post-self-insert-hook #'agl-indent-post-self-insert nil t)
  (agl-flymake-setup))

;; Loaded after the mode definition: these require this file.
(require 'agl-indent)
(require 'agl-flymake)
(require 'agl-run)
(require 'agl-repl)

;; Keybindings live here, with the mode, so the whole surface is visible
;; in one place; each feature file defines only its commands.
(define-key agl-mode-map (kbd "C-c C-c") #'agl-run)
(define-key agl-mode-map (kbd "C-c C-k") #'agl-check)
(define-key agl-mode-map (kbd "C-c C-z") #'agl-repl)
(define-key agl-mode-map (kbd "C-c C-r") #'agl-send-region)
(define-key agl-mode-map (kbd "C-c C-b") #'agl-send-buffer)
(define-key agl-mode-map (kbd "C-c C-l") #'flymake-show-buffer-diagnostics)

(provide 'agl-mode)
;;; agl-mode.el ends here
