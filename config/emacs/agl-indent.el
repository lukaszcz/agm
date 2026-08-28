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
;;   the bracket's content column.
;; - Branch-marker continuation: a line whose first token is `|', `else',
;;   `catch', `until', or `done' continues the enclosing construct and
;;   aligns with the line that opened it.
;; - Block opening: a line that opens a suite indents its body one
;;   `agl-indent-offset' deeper.
;; - Otherwise the previous logical line's indentation carries over.
;;
;; Raw-tail block payloads and multi-line templates are verbatim text, so
;; a line inside one is never re-indented, and backward scans treat those
;; regions as opaque.  Both checks read `syntax-ppss' rather than the
;; buffer text, since the propertize layer is what marks those regions.

;;; Code:

;; `agl-mode.el' requires this file after defining the mode, so requiring it
;; back would be circular; the few helpers used from it are declared instead.
(declare-function agl--ident-boundary-after-p "agl-mode" (pos))
(declare-function agl--ident-boundary-before-p "agl-mode" (pos))
(declare-function agl--operator-name-char-p "agl-mode" (char))
(declare-function agl--operator-token-start-p "agl-mode" (pos))

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
layout rules in docs/agl/reference/lexical-structure.md).")

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

(defconst agl--raw-tail-opener-re
  (concat (regexp-opt '("exec$" "ask$")) "\\(?:::\\[[^]]*\\]\\)?[ \t]*$")
  "Regexp matching a raw-tail opener that carries no inline payload.

The opener spellings end in `$', which is the regexp end-of-line anchor,
so they are escaped through `regexp-opt' rather than spelled inline.")

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

A raw-tail block payload and a multi-line template are significant text,
so a line whose start is already inside one is never re-indented."
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

A raw-tail block payload IS its opener's block, so a line following the
payload returns to the opener's own level instead of indenting under it.")

(defun agl--goto-previous-code-line ()
  "Move to the previous line that participates in layout.

Skips blank and comment-only lines, and skips over verbatim regions so a
raw-tail payload or template body never acts as the previous line.
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

(defun agl--enclosing-bracket-column ()
  "Return the content column of the innermost open bracket, or nil.

While a bracket is open the logical line continues, so a continuation
line aligns just past that bracket."
  (let* ((state (save-excursion (syntax-ppss (line-beginning-position))))
         (open (nth 1 state)))
    (when open
      (save-excursion
        (goto-char open)
        (forward-char 1)
        (skip-chars-forward " \t")
        (if (eolp)
            (+ (progn (goto-char open) (current-indentation)) agl-indent-offset)
          (current-column))))))

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

(defun agl--branch-owner ()
  "Return (INDENT . OPENS-BLOCK) for the construct a branch marker continues.

The marker's own column is whatever the user has typed so far, so it is
not used.  The search starts from the previous code line: when that line
opens a suite it IS the construct's header, and otherwise the header is
the nearest preceding line indented less than it."
  (save-excursion
    (when (agl--goto-previous-code-line)
      (let ((previous (current-indentation)))
        (if (agl--opens-block-p)
            (cons previous t)
          (let ((owner nil))
            (while (and (not owner) (agl--goto-previous-code-line))
              (when (< (current-indentation) previous)
                (setq owner (cons (current-indentation) (agl--opens-block-p)))))
            (or owner (cons 0 nil))))))))

(defun agl--branch-marker-indent ()
  "Return the column the current line's branch marker should sit at.

A `|' introduces a branch inside the construct's body, so it indents one
level under a header that opens a suite.  The terminators `else',
`catch', `until', and `done' close or continue the construct itself and
align with its header."
  (let* ((owner (agl--branch-owner))
         (indent (or (car owner) 0))
         (opens (cdr owner))
         (pipe (save-excursion
                 (beginning-of-line)
                 (skip-chars-forward " \t")
                 (eq (char-after) ?|))))
    (if (and pipe opens) (+ indent agl-indent-offset) indent)))

(defun agl--block-opener-symbol-p (code start)
  "Return non-nil when CODE ends with a symbolic suite introducer.

CODE is the current line\='s code text taken from buffer position START, so
a match\='s index in CODE is also its position in the buffer.  The
introducer must be an operator token of its own: `a->' is one identifier
whose `->' never lexes apart, and the `=' ending `>=' or `!=' continues
that operator instead of assigning."
  (and (string-match agl--block-opener-symbol-re code)
       (let ((pos (+ start (match-beginning 0))))
         (and (agl--operator-token-start-p pos)
              (not (agl--operator-name-char-p (char-before pos)))))))

(defun agl--block-opener-keyword-p (code start)
  "Return non-nil when CODE ends with a keyword suite introducer.

CODE and START are as in `agl--block-opener-symbol-p'.  An identifier
consumes the keyword\='s letters when they merely end a longer name, so the
match counts only where it begins a token (`registry' is not `try')."
  (and (string-match agl--block-opener-keyword-re code)
       (agl--ident-boundary-before-p (+ start (match-beginning 0)))))

(defun agl--raw-tail-opener-p (code start)
  "Return non-nil when CODE is a raw-tail opener carrying no inline payload.

CODE and START are as in `agl--block-opener-symbol-p'; such an opener owns
the indented block that follows it.  `do-exec$' is one identifier rather
than the `exec$' opener, so the same token-boundary check applies."
  (and (string-match agl--raw-tail-opener-re code)
       (agl--ident-boundary-before-p (+ start (match-beginning 0)))))

(defun agl--block-header-p (code start)
  "Return non-nil when CODE is a declaration or compound-statement header.

CODE and START are as in `agl--block-opener-symbol-p'.  A header owns a
block only when its body is not written inline on the same line."
  (and (string-match agl--block-header-re code)
       (agl--ident-boundary-after-p (+ start (match-end 0)))
       (not (string-match-p "=[ \t]*[^ \t]" code))))

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
             (agl--raw-tail-opener-p code start)
             (agl--block-header-p code start)))))

(defun agl-calculate-indent ()
  "Return the column `agl-indent-line' should indent the current line to."
  (save-excursion
    (beginning-of-line)
    (cond
     ;; Inside a bracket the logical line continues.
     ((agl--enclosing-bracket-column))
     ;; A branch marker aligns with the construct it continues.
     ((agl--branch-marker-line-p) (agl--branch-marker-indent))
     (t
      (save-excursion
        (if (not (agl--goto-previous-code-line))
            0
          (let ((previous (current-indentation))
                (crossed agl--crossed-verbatim-region))
            (if (and (agl--opens-block-p) (not crossed))
                (+ previous agl-indent-offset)
              previous))))))))

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
`else\=', `catch\=', `until\=' and `done\=' continue the construct itself rather
than opening a body, so they stay subject to the levels above."
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
          (and (agl--branch-marker-line-p)
               (progn (skip-chars-forward " \t") (eq (char-after) ?|))
               (let ((owner (agl--branch-owner)))
                 (and owner (> column (car owner)))))))))

;; ---------------------------------------------------------------------------
;; Commands
;; ---------------------------------------------------------------------------

(defun agl-indent-line (&optional previous)
  "Indent the current line as AgL code.

With PREVIOUS non-nil (a repeated TAB), cycle to the next candidate
level instead of re-applying the computed one.  A line inside a raw-tail
payload or a multi-line template is left untouched: its text is
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

(defun agl-indent-post-self-insert ()
  "Re-indent the current line after a branch marker is completed.

Typing `|' at the start of a line, or finishing one of the words
`else', `catch', `until', or `done' there, changes which construct the
line belongs to, so the line is re-indented immediately.  The marker is
matched here rather than through a predicate, so the check never depends
on `match-data' surviving another call."
  (when (and (eq major-mode 'agl-mode)
             (not (agl--opaque-line-p))
             (let ((end (point)))
               (save-excursion
                 (beginning-of-line)
                 (skip-chars-forward " \t")
                 (and (looking-at agl--branch-marker-re)
                      (= end (match-end 0))
                      (or (eq (char-after) ?|)
                          (agl--ident-boundary-after-p (match-end 0)))))))
    (agl-indent-line)))

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
