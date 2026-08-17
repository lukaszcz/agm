;;; agl-run.el --- Compile commands for agl-mode -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; Runs an AgL program (`agm exec') or statically checks it (`agm check')
;; through `compile', so diagnostics land in a compilation buffer.  AGM
;; renders diagnostics GNU-style, which the default compilation error
;; regexp already matches, so an error is clickable with no extra entry.

;;; Code:

(require 'compile)

(defcustom agl-exec-command '("agm" "exec")
  "Command vector that runs an AgL program.

The file name is appended when the command runs, so any flags belong
here and nothing beyond the subcommand is hardcoded."
  :type '(repeat string)
  :group 'agl)

(defun agl--compile-command (command file)
  "Return the shell command line running COMMAND on FILE.

Every word is quoted, so a path containing spaces or shell characters
reaches the program unchanged."
  (mapconcat #'shell-quote-argument (append command (list file)) " "))

(defun agl--compile-on-file (command what)
  "Run COMMAND on the current buffer's file through `compile'.

WHAT names the action for the error message raised when the buffer is
not visiting a file."
  (let ((file (buffer-file-name)))
    (unless file
      (user-error "Cannot %s: buffer is not visiting a file" what))
    (save-some-buffers (not compilation-ask-about-save)
                       (lambda () (eq (current-buffer) (get-file-buffer file))))
    (compile (agl--compile-command command file))))

;;;###autoload
(defun agl-run ()
  "Run the current AgL file with `agm exec' in a compilation buffer."
  (interactive)
  (agl--compile-on-file agl-exec-command "run"))

;;;###autoload
(defun agl-check ()
  "Statically check the current AgL file with `agm check'."
  (interactive)
  (agl--compile-on-file agl-check-command "check"))

(provide 'agl-run)
;;; agl-run.el ends here
