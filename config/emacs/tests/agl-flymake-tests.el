;;; agl-flymake-tests.el --- ERT tests for the agl-mode flymake backend -*- lexical-binding: t; -*-

;;; Commentary:

;; Tests the parsing and region mapping of the AgL flymake backend.  The
;; suite never runs `agm' and needs no AGM installation: parsing is a pure
;; function over the checker's output, which is what these tests drive.

;;; Code:

(require 'ert)
(require 'agl-mode)
(require 'agl-flymake)

(defun agl-fm--texts (diagnostics)
  "Return the message text of each diagnostic in DIAGNOSTICS."
  (mapcar #'flymake-diagnostic-text diagnostics))

(defmacro agl-fm--with-buffer (text &rest body)
  "Evaluate BODY in an `agl-mode' buffer holding TEXT and visiting a file."
  (declare (indent 1))
  ;; `agl-flymake-enable' is bound off so entering the mode never starts a
  ;; real `agm check': this suite drives the parsing and mapping directly
  ;; and must not depend on an AGM installation.
  `(let ((file (make-temp-file "agl-flymake-" nil ".agl"))
         (agl-flymake-enable nil))
     (unwind-protect
         (with-temp-buffer
           (setq buffer-file-name file)
           (agl-mode)
           (insert ,text)
           ,@body)
       (delete-file file))))

;; --- Parsing ---

(ert-deftest agl-fm-parses-one-error ()
  (let ((reports (agl-flymake-parse "a.agl:3:5: error: boom\n")))
    (should (= (length reports) 1))
    (let ((report (car reports)))
      (should (equal (agl-flymake-report-path report) "a.agl"))
      (should (= (agl-flymake-report-line report) 3))
      (should (= (agl-flymake-report-column report) 5))
      (should (eq (agl-flymake-report-severity report) :error))
      (should (equal (agl-flymake-report-text report) "boom")))))

(ert-deftest agl-fm-parses-a-warning ()
  (let ((report (car (agl-flymake-parse "a.agl:1:1: warning: careful\n"))))
    (should (eq (agl-flymake-report-severity report) :warning))))

(ert-deftest agl-fm-parses-a-same-line-span ()
  ;; `agm check' prints the span's last column, so the exclusive end is
  ;; one past it.
  (let ((report (car (agl-flymake-parse "a.agl:1:18-31: error: boom\n"))))
    (should (= (agl-flymake-report-column report) 18))
    (should (= (agl-flymake-report-end-column report) 32))
    (should-not (agl-flymake-report-end-line report))))

(ert-deftest agl-fm-parses-a-multi-line-span ()
  (let ((report (car (agl-flymake-parse "a.agl:1:5-3:9: error: boom\n"))))
    (should (= (agl-flymake-report-line report) 1))
    (should (= (agl-flymake-report-column report) 5))
    (should (= (agl-flymake-report-end-line report) 3))
    (should (= (agl-flymake-report-end-column report) 9))))

(ert-deftest agl-fm-parses-a-location-without-a-column ()
  (let ((report (car (agl-flymake-parse "a.agl:7: error: boom\n"))))
    (should (= (agl-flymake-report-line report) 7))
    (should-not (agl-flymake-report-column report))))

(ert-deftest agl-fm-parses-several-diagnostics-in-order ()
  (let ((reports (agl-flymake-parse
                  "a.agl:1:1: error: first\na.agl:2:1: warning: second\n")))
    (should (equal (mapcar #'agl-flymake-report-text reports)
                   '("first" "second")))))

(ert-deftest agl-fm-parses-an-absolute-path ()
  (let ((report (car (agl-flymake-parse "/tmp/x/a.agl:2:3: error: boom\n"))))
    (should (equal (agl-flymake-report-path report) "/tmp/x/a.agl"))))

(ert-deftest agl-fm-ignores-noise-lines ()
  (should-not (agl-flymake-parse "Error: cannot read missing.agl\n"))
  (should-not (agl-flymake-parse "")))

(ert-deftest agl-fm-folds-a-note-into-its-diagnostic ()
  (let ((report (car (agl-flymake-parse
                      "a.agl:1:1: error: boom\n  a.agl:2:1: note: declared here\n"))))
    (should (string-match-p "declared here" (agl-flymake-report-text report)))))

;; --- Region mapping ---

(ert-deftest agl-fm-maps-a-diagnostic-onto-its-line ()
  (agl-fm--with-buffer "let a = 1\nlet b = 2\n"
    (let* ((reports (agl-flymake-parse
                     (format "%s:2:5: error: boom\n" buffer-file-name)))
           (diagnostics (agl-flymake--diagnostics
                         (current-buffer) buffer-file-name reports))
           (diagnostic (car diagnostics)))
      (should (= (length diagnostics) 1))
      (should (>= (flymake-diagnostic-beg diagnostic)
                  (save-excursion (goto-char (point-min))
                                  (forward-line 1)
                                  (point))))
      (should (> (flymake-diagnostic-end diagnostic)
                 (flymake-diagnostic-beg diagnostic))))))

(ert-deftest agl-fm-maps-a-span-to-its-end-column ()
  (agl-fm--with-buffer "let alpha = 1\n"
    (let* ((reports (agl-flymake-parse
                     (format "%s:1:5-9: error: boom\n" buffer-file-name)))
           (diagnostic (car (agl-flymake--diagnostics
                             (current-buffer) buffer-file-name reports))))
      (should (= (flymake-diagnostic-beg diagnostic) 5))
      (should (= (flymake-diagnostic-end diagnostic) 10)))))

(ert-deftest agl-fm-maps-a-column-of-one ()
  (agl-fm--with-buffer "let a = 1\n"
    (let* ((reports (agl-flymake-parse
                     (format "%s:1:1: error: boom\n" buffer-file-name)))
           (diagnostic (car (agl-flymake--diagnostics
                             (current-buffer) buffer-file-name reports))))
      (should (= (flymake-diagnostic-beg diagnostic) 1)))))

(ert-deftest agl-fm-tolerates-a-line-past-the-end-of-the-buffer ()
  (agl-fm--with-buffer "let a = 1\n"
    (let* ((reports (agl-flymake-parse
                     (format "%s:99:3: error: boom\n" buffer-file-name)))
           (diagnostics (agl-flymake--diagnostics
                         (current-buffer) buffer-file-name reports)))
      ;; Whatever it maps to, producing it must not signal.
      (should (listp diagnostics)))))

(ert-deftest agl-fm-attaches-a-foreign-diagnostic-at-point-min ()
  (agl-fm--with-buffer "import lib\n"
    (let* ((reports (agl-flymake-parse "/other/lib.agl:4:2: error: boom\n"))
           (diagnostic (car (agl-flymake--diagnostics
                             (current-buffer) buffer-file-name reports))))
      (should (= (flymake-diagnostic-beg diagnostic) (point-min)))
      (should (string-match-p "/other/lib.agl" (flymake-diagnostic-text diagnostic)))
      (should (string-match-p "boom" (flymake-diagnostic-text diagnostic))))))

(ert-deftest agl-fm-keeps-severity-on-the-diagnostic ()
  (agl-fm--with-buffer "let a = 1\n"
    (let* ((reports (agl-flymake-parse
                     (format "%s:1:1: warning: careful\n" buffer-file-name)))
           (diagnostic (car (agl-flymake--diagnostics
                             (current-buffer) buffer-file-name reports))))
      (should (eq (flymake-diagnostic-type diagnostic) :warning)))))

(ert-deftest agl-fm-empty-output-yields-no-diagnostics ()
  (agl-fm--with-buffer "let a = 1\n"
    (should-not (agl-flymake--diagnostics
                 (current-buffer) buffer-file-name (agl-flymake-parse "")))))

(ert-deftest agl-fm-reports-both-local-and-foreign-diagnostics ()
  (agl-fm--with-buffer "import lib\nlet a = 1\n"
    (let* ((reports (agl-flymake-parse
                     (concat "/other/lib.agl:1:1: error: foreign\n"
                             (format "%s:2:1: error: local\n" buffer-file-name))))
           (diagnostics (agl-flymake--diagnostics
                         (current-buffer) buffer-file-name reports)))
      (should (= (length diagnostics) 2))
      (should (cl-some (lambda (text) (string-match-p "foreign" text))
                       (agl-fm--texts diagnostics)))
      (should (cl-some (lambda (text) (string-match-p "local" text))
                       (agl-fm--texts diagnostics))))))

(ert-deftest agl-fm-foreign-diagnostic-keeps-its-column ()
  (agl-fm--with-buffer "import lib\n"
    (let* ((reports (agl-flymake-parse "/other/lib.agl:4:2: error: boom\n"))
           (diagnostic (car (agl-flymake--diagnostics
                             (current-buffer) buffer-file-name reports))))
      (should (string-match-p "/other/lib.agl:4:2:" (flymake-diagnostic-text diagnostic))))))

(provide 'agl-flymake-tests)
;;; agl-flymake-tests.el ends here
