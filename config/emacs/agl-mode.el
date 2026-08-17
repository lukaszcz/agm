;;; agl-mode.el --- Major mode for AgL source files -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; A major mode for AgL, the statically typed workflow language implemented
;; by AGM (Agent Project Management).  This file provides the package
;; skeleton and the context-sensitive lexing layer: a syntax table plus a
;; `syntax-propertize-function' that make Emacs agree with AgL's lexical
;; rules (see docs/agl/reference/lexical-structure.md), in particular:
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
;; - Raw tails (`exec!'/`ask!', bare or after a `.' projection, with an
;;   optional byte-adjacent `::[T]' type argument) own a verbatim payload:
;;   the rest of the line, or a following indented block.
;;
;; Later files/tasks add font-lock, indentation, imenu/navigation,
;; flymake, and exec/REPL integration.  See
;; docs/agl/reference/lexical-structure.md for the authoritative lexical
;; rules.

;;; Code:

(defgroup agl nil
  "Major mode for editing AgL source files."
  :group 'languages
  :prefix "agl-")

;; ---------------------------------------------------------------------------
;; Keyword inventories (data only; wired into font-lock in a later task).
;;
;; Canonical sources, kept in lockstep with this file and
;; config/micro/agl.yaml -- see the cross-reference comment in
;; src/agm/agl/keywords.py.
;; ---------------------------------------------------------------------------

(defconst agl-keywords
  '("record" "enum" "type" "param" "program" "def" "fn" "let" "var"
    "for" "while" "do" "until" "done" "if" "else" "case" "of" "try" "catch"
    "raise" "return" "break" "continue" "exception" "extends" "builtin"
    "extern" "as" "as?" "and" "or" "not" "is" "in" "to" "downto" "by" "with"
    "true" "false" "null" "infixl" "infixr" "prio")
  "Reserved AgL keywords.

Canonical source: `src/agm/agl/keywords.py' (the `KEYWORDS' frozenset).")

(defconst agl-constant-keywords
  '("true" "false" "null")
  "AgL literal keywords, a subset of `agl-keywords'.")

(defconst agl-soft-keywords
  '("open" "import" "export" "using" "hiding" "scope" "end")
  "AgL soft (contextually promoted) keywords.

Ordinary identifiers outside their promotion window.  Canonical
source: the module/scope soft-keyword table in
`docs/agl/reference/lexical-structure.md', mirrored by
`src/agm/agl/lexer/tokens.py' (OPEN, IMPORT, USING, HIDING, EXPORT,
SCOPE, END).")

(defconst agl-contextual-builtins
  '("print" "ask" "exec")
  "Contextual AgL builtins.

Ordinary NAME tokens given built-in meaning during scope resolution.
Canonical source: `src/agm/agl/lexer/tokens.py'.")

(defconst agl-raw-tail-names
  '("exec!" "ask!")
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
;; boundary.
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
    (let* ((block-end (point))
           (has-content (> block-end block-start)))
      (when has-content
        (put-text-property block-start (1+ block-start)
                            'syntax-table (string-to-syntax "|"))
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

Point is right after a raw-tail opener (`exec!' or `ask!') that
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
  (let ((depth 1))
    (while (and (> depth 0) (not (eobp)))
      (cond
       ((eq (char-after) ?\{) (setq depth (1+ depth)) (forward-char 1))
       ((eq (char-after) ?\}) (setq depth (1- depth)) (forward-char 1))
       (t (forward-char 1))))))

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

A raw-tail opener (`exec!'/`ask!') is only recognized at bracket depth
zero, per the reference's \"only recognized at bracket depth zero\"
rule, so a bracket depth counter is threaded through the scan, seeded
from `(car (syntax-ppss start))' -- safe to call here because
`syntax-propertize' sets `syntax-propertize--done' to END before
invoking this function."
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
            #'agl--propertize-extend-region nil t))

(provide 'agl-mode)
;;; agl-mode.el ends here
