;;; agl-flymake.el --- Flymake backend for agl-mode -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; On-save diagnostics for AgL, backed by `agm check' — the same static
;; pipeline `agm exec' runs, so the editor reports exactly what the
;; compiler does rather than a second, approximate analysis.
;;
;; The check runs on the SAVED file, never on a temporary copy: module
;; roots and imports resolve from the entry file's real path, so a copy
;; elsewhere would resolve a different program.  A modified buffer
;; therefore reports the last state written to disk, which is the
;; standard flymake caveat for on-disk checkers.
;;
;; `agm check' exits 1 whenever it reports a diagnostic, so a non-zero
;; exit is the ordinary path and never a backend failure; only being
;; unable to run the command at all is an error.
;;
;; Diagnostics carry the path of the file they belong to, which for an
;; imported module is not the buffer being checked.  Those foreign
;; diagnostics cannot be placed at a position in this buffer, so they are
;; attached at `point-min' with their path kept in the message.

;;; Code:

(require 'flymake)
(require 'subr-x)

(defcustom agl-check-command '("agm" "check")
  "Command vector that statically checks an AgL file.

The file name is appended when the backend runs it, so any flags belong
here and nothing beyond the subcommand is hardcoded."
  :type '(repeat string)
  :group 'agl)

(defcustom agl-flymake-enable t
  "Whether `agl-mode' turns on `flymake-mode' for AgL buffers."
  :type 'boolean
  :group 'agl)

(defconst agl-flymake--diagnostic-re
  (concat "^\\(.+?\\):\\([0-9]+\\)"
          "\\(?::\\([0-9]+\\)"
          "\\(?:-\\([0-9]+\\)\\(?::\\([0-9]+\\)\\)?\\)?"
          "\\)?"
          ": \\(error\\|warning\\): \\(.*\\)$")
  "Regexp matching one GNU-style diagnostic line from `agm check'.

The location renders as `PATH:LINE', `PATH:LINE:COL',
`PATH:LINE:COL-ENDCOL' for a span within one line, or
`PATH:LINE:COL-ENDLINE:ENDCOL' for a span across lines.  Groups: 1 path,
2 line, 3 column, 4 end column or end line, 5 end column when 4 is an
end line, 6 severity, 7 message.")

(cl-defstruct (agl-flymake-report (:constructor agl-flymake--make-report))
  "One diagnostic parsed from `agm check' output.

PATH, LINE and COLUMN locate it; END-LINE and END-COLUMN bound its span
when the diagnostic carried one; SEVERITY is `:error' or `:warning'; and
TEXT is the human-readable message."
  path line column end-line end-column severity text)

(defun agl-flymake-parse (output)
  "Return the diagnostics `agm check' reported in OUTPUT.

A pure function from the raw process output to a list of
`agl-flymake-report' structures, in the order the checker emitted them.
Lines that are not diagnostics — a note attached to the preceding
diagnostic, or an error with no source location — are folded into the
preceding message or ignored, so unexpected output can never break the
backend."
  (let ((reports nil))
    (dolist (line (split-string (or output "") "\n"))
      (cond
       ((string-match agl-flymake--diagnostic-re line)
        (let* ((column (match-string 3 line))
               (first-end (match-string 4 line))
               (second-end (match-string 5 line))
               (end-line (and second-end (string-to-number first-end)))
               ;; A same-line span prints its last column, one before the
               ;; exclusive end the region needs.
               (end-column (cond (second-end (string-to-number second-end))
                                 (first-end (1+ (string-to-number first-end))))))
          (push (agl-flymake--make-report
                 :path (match-string 1 line)
                 :line (string-to-number (match-string 2 line))
                 :column (and column (string-to-number column))
                 :end-line end-line
                 :end-column end-column
                 :severity (if (equal (match-string 6 line) "warning")
                               :warning
                             :error)
                 :text (match-string 7 line))
                reports)))
       ;; A note continues the diagnostic above it.
       ((and reports (string-match "\\`[ \t]+\\(.*: note: .*\\)\\'" line))
        (let ((report (car reports)))
          (setf (agl-flymake-report-text report)
                (concat (agl-flymake-report-text report)
                        "\n" (string-trim (match-string 1 line))))))))
    (nreverse reports)))

(defun agl-flymake--same-file-p (path file)
  "Return non-nil when PATH and FILE name the same file."
  (and path file
       (or (string= path file)
           (string= (expand-file-name path) (expand-file-name file)))))

(defun agl-flymake--line-count (buffer)
  "Return the number of lines in BUFFER."
  (with-current-buffer buffer
    (save-excursion
      (save-restriction
        (widen)
        (goto-char (point-max))
        (line-number-at-pos)))))

(defun agl-flymake--position (buffer line column)
  "Return the position in BUFFER of LINE and COLUMN, clamped to the buffer.

The checker reads the file from disk, so a buffer edited since the last
save can be shorter than the diagnostics describe; clamping keeps such a
diagnostic visible on the nearest real line instead of signalling."
  (with-current-buffer buffer
    (save-excursion
      (save-restriction
        (widen)
        (goto-char (point-min))
        (forward-line (1- (max 1 (min line (agl-flymake--line-count buffer)))))
        (min (line-end-position)
             (+ (line-beginning-position) (max 0 (1- (or column 1)))))))))

(defun agl-flymake--region (buffer report)
  "Return the (BEG . END) region in BUFFER that REPORT covers.

Positions are clamped to BUFFER, so a diagnostic beyond its end — the
file on disk being longer than the edited buffer — still maps to a real
region rather than signalling."
  (let* ((line (agl-flymake-report-line report))
         (column (agl-flymake-report-column report))
         (end-line (agl-flymake-report-end-line report))
         (end-column (agl-flymake-report-end-column report))
         (beg (agl-flymake--position buffer line column))
         (end (cond
               ((and end-line end-column)
                (agl-flymake--position buffer end-line end-column))
               (end-column (agl-flymake--position buffer line end-column))
               (t nil))))
    (with-current-buffer buffer
      (save-excursion
        (save-restriction
          (widen)
          (unless end
            ;; No span: cover the symbol at BEG, falling back to the rest of
            ;; its line.  The bounds are computed here rather than through
            ;; `flymake-diag-region', which signals for a position past the
            ;; buffer — exactly what a stale on-disk diagnostic names once
            ;; the buffer has been edited shorter.
            (goto-char beg)
            (setq end (or (cdr (bounds-of-thing-at-point 'symbol))
                          (line-end-position)))
            (setq beg (or (car (bounds-of-thing-at-point 'symbol)) beg)))
          (cons beg (min (point-max) (max end (1+ beg)))))))))

(defun agl-flymake--diagnostics (buffer file reports)
  "Return flymake diagnostics for BUFFER, checking FILE, from REPORTS.

A report naming another file — an imported module — has no position in
BUFFER, so it is attached at `point-min' with its path kept in the
message."
  (let ((diagnostics nil))
    (dolist (report reports)
      (let ((type (agl-flymake-report-severity report))
            (text (agl-flymake-report-text report)))
        (if (agl-flymake--same-file-p (agl-flymake-report-path report) file)
            (let ((region (agl-flymake--region buffer report)))
              (when region
                (push (flymake-make-diagnostic buffer (car region) (cdr region)
                                               type text)
                      diagnostics)))
          (with-current-buffer buffer
            (push (flymake-make-diagnostic
                   buffer (point-min) (min (point-max) (1+ (point-min))) type
                   (if (agl-flymake-report-column report)
                       (format "%s:%s:%s: %s"
                               (agl-flymake-report-path report)
                               (agl-flymake-report-line report)
                               (agl-flymake-report-column report)
                               text)
                     (format "%s:%s: %s"
                             (agl-flymake-report-path report)
                             (agl-flymake-report-line report)
                             text)))
                  diagnostics)))))
    (nreverse diagnostics)))

(defvar-local agl-flymake--process nil
  "The `agm check' process currently checking this buffer, if any.")

(defun agl-flymake-backend (report-fn &rest _args)
  "Check this buffer's file with `agm check' and pass results to REPORT-FN.

Registered on `flymake-diagnostic-functions'.  A check already running
for this buffer is cancelled first, and a result is reported only while
the process that produced it is still the current one, so a stale run
can never overwrite a newer one."
  (unless (executable-find (car agl-check-command))
    (error "Cannot find %s" (car agl-check-command)))
  (let ((file (buffer-file-name)))
    (unless file
      (error "Buffer is not visiting a file"))
    (when (process-live-p agl-flymake--process)
      (kill-process agl-flymake--process))
    (let* ((buffer (current-buffer))
           (output (generate-new-buffer " *agl-check*"))
           (process
            (make-process
             :name "agl-check"
             :noquery t
             :connection-type 'pipe
             :buffer output
             :command (append agl-check-command (list file))
             :sentinel
             (lambda (proc _event)
               (unless (process-live-p proc)
                 (unwind-protect
                     (when (with-current-buffer buffer
                             (eq proc agl-flymake--process))
                       ;; `agm check' exits 1 whenever it reports a
                       ;; diagnostic, so the exit status says nothing about
                       ;; whether the run succeeded.
                       (let ((text (with-current-buffer output
                                     (buffer-string))))
                         (funcall report-fn
                                  (agl-flymake--diagnostics
                                   buffer file (agl-flymake-parse text)))))
                   (kill-buffer output)))))))
      (setq agl-flymake--process process))))

(defun agl-flymake-setup ()
  "Register the AgL backend and enable `flymake-mode' when configured.

The backend checks the file on disk, so `flymake-mode' is turned on only
for a buffer that visits one."
  (add-hook 'flymake-diagnostic-functions #'agl-flymake-backend nil t)
  (when (and agl-flymake-enable (buffer-file-name))
    (flymake-mode 1)))

(provide 'agl-flymake)
;;; agl-flymake.el ends here
