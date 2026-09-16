;;; agl-run-tests.el --- ERT tests for agl-mode compile commands -*- lexical-binding: t; -*-

;;; Commentary:

;; Tests the command lines `agl-run' and `agl-check' hand to `compile',
;; and that AGM's GNU-style diagnostics are recognized by the default
;; compilation error regexp.  No process is ever spawned: `compile' is
;; stubbed.

;;; Code:

(require 'ert)
(require 'agl-mode)
(require 'agl-run)

(defvar agl-run-tests--command nil
  "The command line the stubbed `compile' was called with.")

(defmacro agl-run--with-stubbed-compile (file &rest body)
  "Run BODY in an `agl-mode' buffer visiting FILE, with `compile' stubbed."
  (declare (indent 1))
  `(let ((agl-run-tests--command nil)
         (agl-flymake-enable nil))
     (cl-letf (((symbol-function 'compile)
                (lambda (command &rest _) (setq agl-run-tests--command command)))
               ((symbol-function 'save-some-buffers) (lambda (&rest _) nil)))
       (with-temp-buffer
         (setq buffer-file-name ,file)
         (agl-mode)
         ,@body))))

;; --- Command construction ---

(ert-deftest agl-run-builds-the-exec-command ()
  (agl-run--with-stubbed-compile "/tmp/work/flow.agl"
    (agl-run)
    (should (equal agl-run-tests--command "agm exec /tmp/work/flow.agl"))))

(ert-deftest agl-run-builds-the-check-command ()
  (agl-run--with-stubbed-compile "/tmp/work/flow.agl"
    (agl-check)
    (should (equal agl-run-tests--command "agm check /tmp/work/flow.agl"))))

(ert-deftest agl-run-quotes-a-path-needing-it ()
  (agl-run--with-stubbed-compile "/tmp/my work/a file.agl"
    (agl-run)
    ;; The quoted command must still name the original path.
    (should (string-match-p "work" agl-run-tests--command))
    (should-not (string-match-p " work/a file" agl-run-tests--command))))

(ert-deftest agl-run-honours-a-customized-exec-command ()
  (let ((agl-exec-command '("agm" "exec" "--no-stdlib")))
    (agl-run--with-stubbed-compile "/tmp/flow.agl"
      (agl-run)
      (should (equal agl-run-tests--command "agm exec --no-stdlib /tmp/flow.agl")))))

(ert-deftest agl-run-honours-a-customized-check-command ()
  (let ((agl-check-command '("agm" "check" "-I" "/lib")))
    (agl-run--with-stubbed-compile "/tmp/flow.agl"
      (agl-check)
      (should (equal agl-run-tests--command "agm check -I /lib /tmp/flow.agl")))))

(ert-deftest agl-run-without-a-file-signals ()
  (let ((agl-flymake-enable nil))
    (with-temp-buffer
      (agl-mode)
      (should-error (agl-run) :type 'user-error))))

(ert-deftest agl-mode-provides-an-agl-menu ()
  (let ((agl-flymake-enable nil))
    (with-temp-buffer
      (agl-mode)
      (should (lookup-key agl-mode-map [menu-bar agl])))))

(ert-deftest agl-mode-binds-reload-to-control-c-control-r ()
  (should (eq (lookup-key agl-mode-map (kbd "C-c C-r"))
              #'agl-repl-reload-buffer))
  (should (eq (lookup-key agl-mode-map (kbd "C-c C-s"))
              #'agl-send-region)))

;; --- AGM diagnostics are clickable in a compilation buffer ---

(ert-deftest agl-run-compilation-recognizes-agm-diagnostics ()
  ;; AGM renders `path:line:col: error: message', which the default
  ;; compilation regexps already match -- no mode-specific entry needed.
  (with-temp-buffer
    (let ((line "/tmp/work/flow.agl:3:5: error: 'x' is not defined."))
      (should (cl-some (lambda (entry)
                         (let ((regexp (if (symbolp (car entry))
                                           (car (cdr (assq (car entry)
                                                           compilation-error-regexp-alist-alist)))
                                         (car entry))))
                           (and (stringp regexp) (string-match-p regexp line))))
                       (mapcar (lambda (key)
                                 (or (assq key compilation-error-regexp-alist-alist)
                                     (list key)))
                               compilation-error-regexp-alist))))))

(ert-deftest agl-run-compilation-recognizes-a-span-diagnostic ()
  (let ((line "/tmp/work/flow.agl:1:18-31: error: boom"))
    (should (cl-some (lambda (key)
                       (let ((entry (assq key compilation-error-regexp-alist-alist)))
                         (and entry (stringp (nth 1 entry))
                              (string-match-p (nth 1 entry) line))))
                     compilation-error-regexp-alist))))

(provide 'agl-run-tests)
;;; agl-run-tests.el ends here
