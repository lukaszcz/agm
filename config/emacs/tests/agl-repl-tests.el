;;; agl-repl-tests.el --- ERT tests for the inferior AgL REPL -*- lexical-binding: t; -*-

;;; Commentary:

;; Tests the inferior-REPL command construction, what `agl-send-region'
;; and `agl-send-buffer' transmit, and the prompt regexp.  No process is
;; ever spawned: the comint entry points are stubbed.

;;; Code:

(require 'ert)
(require 'agl-mode)
(require 'agl-repl)

(defvar agl-repl-tests--spawn nil
  "The (PROGRAM . ARGS) the stubbed `make-comint-in-buffer' received.")

(defvar agl-repl-tests--sent nil
  "The strings the stubbed `comint-send-string' received, in order.")

(defvar agl-repl-tests--connection-type nil
  "The process connection type the REPL selected.")

(defmacro agl-repl--with-stubs (&rest body)
  "Run BODY with the comint process entry points stubbed out."
  (declare (indent 0))
  `(let ((agl-repl-tests--spawn nil)
         (agl-repl-tests--sent nil)
         (agl-repl-tests--connection-type nil)
         (agl-flymake-enable nil)
         (agl-repl-buffer-name "*AgL REPL test*"))
     (unwind-protect
         (cl-letf (((symbol-function 'make-comint-in-buffer)
                    (lambda (_name buffer program _startfile &rest args)
                      (setq agl-repl-tests--spawn (cons program args))
                      (setq agl-repl-tests--connection-type process-connection-type)
                      buffer))
                   ((symbol-function 'comint-check-proc) (lambda (&rest _) nil))
                   ((symbol-function 'get-buffer-process) (lambda (&rest _) 'stub))
                   ((symbol-function 'comint-send-string)
                    (lambda (_process text)
                      (setq agl-repl-tests--sent
                            (append agl-repl-tests--sent (list text))))))
           ,@body)
       (when (get-buffer agl-repl-buffer-name)
         (kill-buffer agl-repl-buffer-name)))))

;; --- Process construction ---

(ert-deftest agl-repl-spawns-the-configured-command ()
  (agl-repl--with-stubs
    (agl-repl-buffer)
    (should (equal agl-repl-tests--spawn '("agm" "repl")))
    (should agl-repl-tests--connection-type)))

(ert-deftest agl-repl-honours-a-customized-command ()
  (let ((agl-repl-command '("agm" "repl" "--plain" "--quiet")))
    (agl-repl--with-stubs
      (agl-repl-buffer)
      (should (equal agl-repl-tests--spawn '("agm" "repl" "--plain" "--quiet"))))))

(ert-deftest agl-repl-reuses-one-buffer ()
  (agl-repl--with-stubs
    (should (eq (agl-repl-buffer) (agl-repl-buffer)))))

(ert-deftest agl-repl-renders-ansi-output ()
  (with-temp-buffer
    (agl-repl-mode)
    (should (memq #'ansi-color-process-output comint-output-filter-functions))))

;; --- Sending source ---

(ert-deftest agl-repl-send-region-sends-the-text-with-a-newline ()
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "let x = 1")
      (agl-send-region (point-min) (point-max)))
    (should (equal agl-repl-tests--sent '("let x = 1\n")))))

(ert-deftest agl-repl-send-buffer-sends-everything ()
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "let x = 1\nlet y = 2\n")
      (agl-send-buffer))
    (should (equal agl-repl-tests--sent '("let x = 1\nlet y = 2\n")))))

(ert-deftest agl-repl-reload-buffer-resets-before-sending-source ()
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "let x = 1")
      (agl-repl-reload-buffer))
    (should (equal agl-repl-tests--sent '(":reset\n" "let x = 1\n")))))

(ert-deftest agl-repl-reload-on-save-is-opt-in ()
  (let ((agl-repl-reload-on-save t))
    (agl-repl--with-stubs
      (with-temp-buffer
        (insert "let x = 1")
        (agl-mode)
        (run-hooks 'after-save-hook))
      (should (equal agl-repl-tests--sent '(":reset\n" "let x = 1\n"))))))

(ert-deftest agl-repl-sends-a-multi-line-block-unsplit ()
  ;; The REPL accumulates continuation lines until an entry is complete, so a
  ;; block is sent as one string rather than split here.
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "def f() -> int =\n  1\n")
      (agl-send-buffer))
    (should (= (length agl-repl-tests--sent) 1))
    (should (equal (car agl-repl-tests--sent) "def f() -> int =\n  1\n\n"))))

(ert-deftest agl-repl-terminates-a-block-left-open-by-its-last-line ()
  ;; An indented last line leaves the reader inside a layout block, which can
  ;; always take one more line; the blank line is what closes it.  A single
  ;; final newline would only start another continuation line.
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "case 1 of\n  | 1 => print(\"one\")\n  | _ => print(\"other\")\n")
      (agl-send-buffer))
    (should (equal agl-repl-tests--sent
                   '("case 1 of\n  | 1 => print(\"one\")\n  | _ => print(\"other\")\n\n")))))

(ert-deftest agl-repl-terminates-a-trailing-raw-tail-block ()
  ;; An indented raw-tail payload is the same case: its blank line is what
  ;; tells the REPL reader the payload is complete.
  (agl-repl--with-stubs
    (dolist (opener '("exec$" "ask$"))
      (with-temp-buffer
        (agl-mode)
        (insert opener "\n  payload\n")
        (agl-send-buffer)))
    (should (equal agl-repl-tests--sent
                   '("exec$\n  payload\n\n" "ask$\n  payload\n\n")))))

(ert-deftest agl-repl-does-not-terminate-a-closed-region ()
  ;; Sibling statements at column zero are each a complete entry, so the
  ;; reader needs no terminator and one is not sent.
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "let x = 1\nlet y = 2\n")
      (agl-send-buffer))
    (should (equal agl-repl-tests--sent '("let x = 1\nlet y = 2\n")))))

(ert-deftest agl-repl-does-not-double-a-trailing-newline ()
  (agl-repl--with-stubs
    (with-temp-buffer
      (insert "let x = 1\n\n")
      (agl-send-buffer))
    (should (equal agl-repl-tests--sent '("let x = 1\n")))))

;; --- Prompt regexp ---

(ert-deftest agl-repl-prompt-regexp-matches-both-prompts ()
  (should (string-match-p agl-repl-prompt-regexp "agl> "))
  (should (string-match-p agl-repl-prompt-regexp "...> ")))

(ert-deftest agl-repl-prompt-regexp-rejects-output-lines ()
  (should-not (string-match-p agl-repl-prompt-regexp "x : int = 3"))
  (should-not (string-match-p agl-repl-prompt-regexp "  agl> indented"))
  (should-not (string-match-p agl-repl-prompt-regexp "error: boom")))

(provide 'agl-repl-tests)
;;; agl-repl-tests.el ends here
