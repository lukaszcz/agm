;;; agl-repl.el --- Inferior AgL REPL for agl-mode -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; An inferior AgL REPL over comint.  Its pty preserves the REPL's own
;; prompt_toolkit ANSI styling, which comint renders in the buffer.

;;; Code:

(require 'ansi-color)
(require 'comint)

(defcustom agl-repl-command '("agm" "repl")
  "Command vector that starts the inferior AgL REPL.

The default runs the REPL's rich terminal front end so its ANSI syntax
highlighting reaches comint."
  :type '(repeat string)
  :group 'agl)

(defcustom agl-repl-buffer-name "*AgL REPL*"
  "Name of the buffer running the inferior AgL REPL."
  :type 'string
  :group 'agl)

(defcustom agl-repl-reload-on-save nil
  "Whether saving an AgL buffer resets and reloads the inferior REPL.

Reloading starts from a clean session, so deleted declarations disappear as
well as changed ones taking effect.  It deliberately remains opt-in because
it also discards expressions entered manually at the REPL prompt."
  :type 'boolean
  :group 'agl)

(defconst agl-repl-prompt-regexp "^\\(?:agl> \\|\\.\\.\\.> \\)"
  "Regexp matching the REPL's primary and continuation prompts.

Comint removes ANSI sequences before using this regexp.  The continuation
prompt follows the previous entry on the same line, so in practice only the
primary prompt matches at line start; the alternative is kept so the regexp
describes both.")

(define-derived-mode agl-repl-mode comint-mode "AgL-REPL"
  "Major mode for an inferior AgL REPL."
  (ansi-color-for-comint-mode-on)
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
      (let ((process-connection-type t))
        (apply #'make-comint-in-buffer "AgL REPL" buffer
               (car agl-repl-command) nil (cdr agl-repl-command)))
      (with-current-buffer buffer (agl-repl-mode)))
    buffer))

;;;###autoload
(defun agl-repl ()
  "Start the inferior AgL REPL if needed and switch to its buffer."
  (interactive)
  (pop-to-buffer (agl-repl-buffer)))

(defun agl-repl--open-block-p (text)
  "Return non-nil when TEXT leaves the REPL reader inside a block.

The REPL keeps an entry open while its latest line is indented, since a layout
block accepts one more line however well what precedes it parses.  An indented
raw-tail payload is the same case, and is closed by the same blank line."
  (let* ((lines (split-string (string-trim-right text) "\n"))
         (last (car (last lines))))
    (and (cdr lines) (string-match-p "\\`[ \t]" last))))

(defun agl-repl-send-string (text)
  "Send TEXT to the inferior AgL REPL, followed by a newline.

The REPL reads continuation lines until an entry is complete, so a
multi-line TEXT is sent unchanged rather than split into entries here.
A TEXT that leaves a block open (`agl-repl--open-block-p') is followed by
a blank line instead, which is what closes that block."
  (let ((buffer (agl-repl-buffer)))
    (comint-send-string (get-buffer-process buffer)
                        (concat (string-trim-right text "\n+")
                                (if (agl-repl--open-block-p text) "\n\n" "\n")))
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

;;;###autoload
(defun agl-repl-reload-buffer ()
  "Reset the inferior REPL and load the current buffer into it.

The reset makes reload faithful to the buffer: declarations removed from the
source cannot survive as stale REPL state."
  (interactive)
  (agl-repl-send-string ":reset")
  (agl-send-buffer))

(defun agl-repl--reload-after-save ()
  "Reload the current buffer when `agl-repl-reload-on-save' is enabled."
  (when agl-repl-reload-on-save
    (agl-repl-reload-buffer)))

(defun agl-repl-setup-reload-on-save ()
  "Install this buffer's optional REPL reload-on-save hook."
  (add-hook 'after-save-hook #'agl-repl--reload-after-save nil t))

(provide 'agl-repl)
;;; agl-repl.el ends here
