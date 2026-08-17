;;; agl-repl.el --- Inferior AgL REPL for agl-mode -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; An inferior AgL REPL over comint, running `agm repl --plain'.  Plain
;; mode prints unstyled prompts and reads entries line by line, which is
;; what makes the session drivable from a comint buffer; it also
;; accumulates continuation lines until an entry is complete, so a
;; multi-line region can be sent as-is rather than split here.

;;; Code:

(require 'comint)

(defcustom agl-repl-command '("agm" "repl" "--plain")
  "Command vector that starts the inferior AgL REPL.

`--plain' is explicit even though the REPL auto-detects a non-terminal,
so the front end never depends on how the process was spawned."
  :type '(repeat string)
  :group 'agl)

(defcustom agl-repl-buffer-name "*AgL REPL*"
  "Name of the buffer running the inferior AgL REPL."
  :type 'string
  :group 'agl)

(defconst agl-repl-prompt-regexp "^\\(?:agl> \\|\\.\\.\\.> \\)"
  "Regexp matching the plain REPL's primary and continuation prompts.

The spellings come from the REPL's own prompt constants, which the plain
front end prints unstyled.")

(define-derived-mode agl-repl-mode comint-mode "AgL-REPL"
  "Major mode for an inferior AgL REPL."
  (setq-local comint-prompt-regexp agl-repl-prompt-regexp)
  (setq-local comint-prompt-read-only t)
  (setq-local comint-process-echoes nil))

(defun agl-repl-process ()
  "Return the live inferior AgL REPL process, or nil."
  (let ((buffer (get-buffer agl-repl-buffer-name)))
    (and buffer (get-buffer-process buffer))))

(defun agl-repl-buffer ()
  "Return the inferior AgL REPL buffer, starting the process if needed."
  (let ((buffer (get-buffer-create agl-repl-buffer-name)))
    (unless (comint-check-proc buffer)
      (apply #'make-comint-in-buffer "AgL REPL" buffer
             (car agl-repl-command) nil (cdr agl-repl-command))
      (with-current-buffer buffer (agl-repl-mode)))
    buffer))

;;;###autoload
(defun agl-repl ()
  "Start the inferior AgL REPL if needed and switch to its buffer."
  (interactive)
  (pop-to-buffer (agl-repl-buffer)))

(defun agl-repl-send-string (text)
  "Send TEXT to the inferior AgL REPL, followed by a newline.

The REPL reads continuation lines until an entry is complete, so a
multi-line TEXT is sent unchanged rather than split into entries here."
  (let ((buffer (agl-repl-buffer)))
    (comint-send-string (get-buffer-process buffer)
                        ;; Exactly one terminating newline: trailing blank
                        ;; lines would submit extra empty entries.
                        (concat (string-trim-right text "\n+") "\n"))
    buffer))

;;;###autoload
(defun agl-send-region (start end)
  "Send the region between START and END to the inferior AgL REPL."
  (interactive "r")
  (agl-repl-send-string (buffer-substring-no-properties start end)))

;;;###autoload
(defun agl-send-buffer ()
  "Send the whole buffer to the inferior AgL REPL."
  (interactive)
  (agl-send-region (point-min) (point-max)))

(provide 'agl-repl)
;;; agl-repl.el ends here
