;;; agl-repl.el --- Inferior AgL REPL for agl-mode -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; An inferior AgL REPL over comint.  Comint advertises a dumb terminal, so the
;; integration runs the plain REPL and applies AgL font-lock to prompt input.

;;; Code:

(require 'ansi-color)
(require 'cl-lib)
(require 'comint)

(declare-function agl--setup-font-lock "agl-mode"
                  (&optional fontify-region-function))

(defcustom agl-repl-command '("agm" "repl" "--plain")
  "Command vector that starts the inferior AgL REPL.

The default uses the line-oriented frontend; `agl-repl-mode' provides native
syntax highlighting without requiring terminal emulation."
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

(defvar-local agl-repl--sent-input-echo ""
  "Terminal input echo still expected from programmatic REPL sends.")

(defvar-local agl-repl--sent-prompt-count 0
  "REPL prompts still expected from programmatic REPL sends.")

(defconst agl-repl--sent-prompt-regexp
  "\\(?:\e\\[[0-9;]*m\\)*\\(?:agl> \\|\\.\\.\\.> \\)\\(?:\e\\[[0-9;]*m\\)*"
  "Regexp matching a plain or styled REPL prompt in process output.")

(defun agl-repl--filter-sent-prompts (output)
  "Remove prompts emitted while source is injected into the REPL."
  (let ((start 0)
        (visible nil))
    (while (and (> agl-repl--sent-prompt-count 0)
                (string-match agl-repl--sent-prompt-regexp output start))
      (push (substring output start (match-beginning 0)) visible)
      (setq start (match-end 0))
      (cl-decf agl-repl--sent-prompt-count))
    (concat (apply #'concat (nreverse visible)) (substring output start))))

(defun agl-repl--filter-sent-echo (output)
  "Remove terminal echo from OUTPUT while retaining REPL results."
  (let ((echo agl-repl--sent-input-echo)
        (start 0)
        (visible nil)
        (done nil))
    (while (and (not (string-empty-p echo)) (not done))
      (let* ((line-end (string-match "\r\n" echo))
             (line (substring echo 0 (+ line-end 2)))
             (match (string-match (regexp-quote line) output start)))
        (if match
            (progn
              (push (substring output start match) visible)
              (setq start (+ match (length line)))
              (setq echo (substring echo (length line))))
          (when (string-prefix-p (substring output start) line)
            (setq echo (concat (substring line (- (length output) start))
                               (substring echo (length line))))
            (setq start (length output)))
          (setq done t))))
    (setq agl-repl--sent-input-echo echo)
    (concat (apply #'concat (nreverse visible)) (substring output start))))

(defun agl-repl--filter-sent-input (output)
  "Remove terminal echo and prompts from `agl-repl-send-string' source."
  (let ((visible (agl-repl--filter-sent-prompts
                  (agl-repl--filter-sent-echo output))))
    ;; Prompt-toolkit disables terminal echo.  A visible final prompt therefore
    ;; completes the send even when none of the expected echo was observed.
    (when (and (= agl-repl--sent-prompt-count 0)
               (string-match-p agl-repl--sent-prompt-regexp visible))
      (setq agl-repl--sent-input-echo ""))
    visible))

(defun agl-repl--fontify-region (start end loudly)
  "Fontify AgL input after REPL prompts between START and END.

LOUDLY is forwarded to `font-lock-default-fontify-region'.  Output remains
unstyled unless it carries styling of its own."
  (save-excursion
    (goto-char start)
    (beginning-of-line)
    (while (re-search-forward agl-repl-prompt-regexp end t)
      (let ((code-start (point))
            (code-end (line-end-position)))
        (when (< code-start code-end)
          (font-lock-default-unfontify-region code-start code-end)
          (save-restriction
            (narrow-to-region code-start code-end)
            (syntax-propertize (point-max))
            (font-lock-default-fontify-region code-start code-end loudly)))))))

(defun agl-repl--setup-input-buffer ()
  "Set up the indirect buffer used to fontify REPL input."
  (prog-mode)
  (agl--setup-font-lock))

(define-derived-mode agl-repl-mode comint-mode "AgL-REPL"
  "Major mode for an inferior AgL REPL."
  (setq-local comint-highlight-input nil)
  (if (fboundp 'comint-fontify-input-mode)
      (progn
        (setq-local comint-indirect-setup-function #'agl-repl--setup-input-buffer)
        (comint-fontify-input-mode 1))
    (agl--setup-font-lock #'agl-repl--fontify-region))
  (ansi-color-for-comint-mode-on)
  (setq-local comint-prompt-regexp agl-repl-prompt-regexp)
  (setq-local comint-prompt-read-only t)
  (setq-local comint-process-echoes nil)
  (setq-local agl-repl--sent-prompt-count 0)
  (add-hook 'comint-preoutput-filter-functions #'agl-repl--filter-sent-input nil t))

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
`$' verbatim block payload is the same case, and is closed by the same blank
line."
  (let* ((lines (split-string (string-trim-right text) "\n"))
         (last (car (last lines))))
    (and (cdr lines) (string-match-p "\\`[ \t]" last))))

(defun agl-repl--sent-text (text)
  "Return the complete terminal input sent for REPL TEXT."
  (concat (string-trim-right text "\n+")
          (if (agl-repl--open-block-p text) "\n\n" "\n")))

(defun agl-repl-send-string (text &optional suppress-final-prompt)
  "Send TEXT to the inferior AgL REPL, followed by a newline.

The REPL reads continuation lines until an entry is complete, so a
multi-line TEXT is sent unchanged rather than split into entries here.
A TEXT that leaves a block open (`agl-repl--open-block-p') is followed by
a blank line instead, which is what closes that block.  When
SUPPRESS-FINAL-PROMPT is non-nil, hide the prompt after the final entry too."
  (let ((buffer (agl-repl-buffer))
        (sent-text (agl-repl--sent-text text)))
    (with-current-buffer buffer
      ;; The pty expands newlines while echoing its input before the REPL emits
      ;; a result.  Hide intermediate prompts but preserve the final prompt so
      ;; the REPL is visibly ready for manually typed input.
      (let ((prompt-count (cl-count ?\n sent-text)))
        (setq agl-repl--sent-input-echo
              (concat agl-repl--sent-input-echo
                      (replace-regexp-in-string "\n" "\r\n" sent-text)))
        (cl-incf agl-repl--sent-prompt-count
                 (if suppress-final-prompt prompt-count (1- prompt-count)))))
    (comint-send-string (get-buffer-process buffer) sent-text)
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
  (agl-repl-send-string ":reset" t)
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
