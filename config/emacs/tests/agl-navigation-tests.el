;;; agl-navigation-tests.el --- ERT tests for agl-mode imenu/defun motion -*- lexical-binding: t; -*-

;;; Commentary:

;; Tests `agl-mode''s declaration navigation: the imenu index (Programs,
;; Functions, Types, Scopes, with scope-nested declarations indexed under
;; a qualified name) and `beginning-of-defun'/`end-of-defun' motion over
;; top-level declarations.

;;; Code:

(require 'ert)
(require 'agl-mode)

(defmacro agl-nav--with-buffer (text &rest body)
  "Evaluate BODY in a temporary `agl-mode' buffer holding TEXT."
  (declare (indent 1))
  `(with-temp-buffer
     (agl-mode)
     (insert ,text)
     (font-lock-ensure)
     (goto-char (point-min))
     ,@body))

(defun agl-nav--pos-after (needle)
  "Return the buffer position right after the first occurrence of NEEDLE."
  (save-excursion
    (goto-char (point-min))
    (search-forward needle)
    (point)))

(defun agl-nav--pos-before (needle)
  "Return the buffer position of the first occurrence of NEEDLE."
  (save-excursion
    (goto-char (point-min))
    (search-forward needle)
    (- (point) (length needle))))

(defun agl-nav--category-names (index category)
  "Return the entry names of CATEGORY in imenu INDEX."
  (mapcar #'car (cdr (assoc category index))))

;; --- imenu index ---

(ert-deftest agl-nav-imenu-programs-category ()
  (agl-nav--with-buffer "program def main() -> unit =\n  print \"hi\"\n"
    (let ((index (agl-imenu-create-index)))
      (should (member "main" (agl-nav--category-names index "Programs")))
      (should-not (member "main" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-functions-category ()
  (agl-nav--with-buffer
      (concat "def greet(x: text) -> text = x\n"
              "extern def to_slug(title: text) -> text\n")
    (let ((index (agl-imenu-create-index)))
      (should (member "greet" (agl-nav--category-names index "Functions")))
      (should (member "to_slug" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-operator-named-def-is-indexed ()
  ;; A declaration may name an operator; the index has to reach it under the
  ;; name it is spelled with.
  (agl-nav--with-buffer "def |>[A, B](x: A, f: fn(A) -> B) -> B = f(x)\n"
    (let ((index (agl-imenu-create-index)))
      (should (member "|>" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-types-category ()
  (agl-nav--with-buffer
      (concat "record Point(x: int, y: int)\n"
              "enum Review\n  | Pass\n  | Fail\n"
              "type Status = Review\n"
              "exception Retryable extends Exception\n")
    (let ((index (agl-imenu-create-index)))
      (dolist (name '("Point" "Review" "Status" "Retryable"))
        (should (member name (agl-nav--category-names index "Types")))))))

(ert-deftest agl-nav-imenu-scopes-category ()
  (agl-nav--with-buffer "scope Geometry\n  def area() -> int = 0\nend Geometry\n"
    (let ((index (agl-imenu-create-index)))
      (should (member "Geometry" (agl-nav--category-names index "Scopes"))))))

(ert-deftest agl-nav-imenu-nested-scope-method-is-qualified ()
  (agl-nav--with-buffer
      (concat "scope Point\n"
              "  def distance(self) -> int = 0\n"
              "end Point\n")
    (let ((index (agl-imenu-create-index)))
      (should (member "Point::distance" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-decl-path-method-is-qualified ()
  (agl-nav--with-buffer "def Box::get[E](self) -> E = self.value\n"
    (let ((index (agl-imenu-create-index)))
      (should (member "Box::get" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-entry-marker-points-at-declared-name ()
  (agl-nav--with-buffer "def greet(x: text) -> text = x\n"
    (let* ((index (agl-imenu-create-index))
           (entry (assoc "greet" (cdr (assoc "Functions" index)))))
      (should entry)
      (should (= (marker-position (cdr entry)) (agl-nav--pos-after "def "))))))

;; --- beginning-of-defun / end-of-defun ---

(ert-deftest agl-nav-beginning-of-defun-moves-to-enclosing-declaration ()
  (agl-nav--with-buffer
      (concat "def first() -> int =\n"
              "  1\n"
              "def second() -> int =\n"
              "  2\n")
    (goto-char (agl-nav--pos-after "second() -> int =\n  "))
    (beginning-of-defun)
    (should (= (point) (agl-nav--pos-after "1\n")))))

(ert-deftest agl-nav-beginning-of-defun-repeated-call-moves-further-back ()
  (agl-nav--with-buffer
      (concat "def first() -> int =\n"
              "  1\n"
              "def second() -> int =\n"
              "  2\n")
    (goto-char (agl-nav--pos-after "1\n"))
    (beginning-of-defun)
    (should (= (point) (point-min)))))

(ert-deftest agl-nav-end-of-defun-moves-to-next-toplevel-declaration ()
  (agl-nav--with-buffer
      (concat "def first() -> int =\n"
              "  1\n"
              "def second() -> int =\n"
              "  2\n")
    (goto-char (point-min))
    (end-of-defun)
    (should (= (point) (agl-nav--pos-after "1\n")))))

(ert-deftest agl-nav-end-of-defun-at-last-declaration-reaches-buffer-end ()
  (agl-nav--with-buffer "def only() -> int =\n  1\n"
    (goto-char (point-min))
    (end-of-defun)
    (should (= (point) (point-max)))))

;; --- imenu: builtin def, multiple program defs, multi-segment scope,
;;     and scope-stack pop ---

(ert-deftest agl-nav-imenu-builtin-def-is-functions-category ()
  (agl-nav--with-buffer "builtin def copy[T](value: T) -> T\n"
    (let ((index (agl-imenu-create-index)))
      (should (member "copy" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-multiple-program-defs ()
  (agl-nav--with-buffer
      (concat "program def main() -> unit =\n  print \"hi\"\n"
              "program def other() -> unit =\n  print \"bye\"\n")
    (let ((index (agl-imenu-create-index)))
      (should (member "main" (agl-nav--category-names index "Programs")))
      (should (member "other" (agl-nav--category-names index "Programs"))))))

(ert-deftest agl-nav-imenu-multi-segment-scope-is-qualified ()
  (agl-nav--with-buffer
      (concat "scope A::B\n"
              "  def f() -> int = 0\n"
              "end A::B\n")
    (let ((index (agl-imenu-create-index)))
      (should (member "A::B" (agl-nav--category-names index "Scopes")))
      (should (member "A::B::f" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-declaration-after-end-is-not-qualified ()
  (agl-nav--with-buffer
      (concat "scope Geometry\n"
              "  def area() -> int = 0\n"
              "end Geometry\n"
              "def outside() -> int = 1\n")
    (let ((index (agl-imenu-create-index)))
      (should (member "outside" (agl-nav--category-names index "Functions")))
      (should-not (member "Geometry::outside" (agl-nav--category-names index "Functions"))))))

;; --- imenu is case-sensitive ---

(ert-deftest agl-nav-imenu-case-sensitive-uppercase-def-not-indexed ()
  (agl-nav--with-buffer "DEF loud() -> int = 0\ndef quiet() -> int = 1\n"
    (let ((index (agl-imenu-create-index)))
      (should-not (member "loud" (agl-nav--category-names index "Functions")))
      (should (member "quiet" (agl-nav--category-names index "Functions"))))))

;; --- imenu/defun ignore declarations that are only text ---

(ert-deftest agl-nav-imenu-skips-commented-out-declaration ()
  (agl-nav--with-buffer "# def hidden() -> int = 0\ndef real() -> int = 1\n"
    (let ((index (agl-imenu-create-index)))
      (should-not (member "hidden" (agl-nav--category-names index "Functions")))
      (should (member "real" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-skips-declaration-inside-verbatim-payload ()
  (agl-nav--with-buffer "exec $ echo def hidden() -> int = 0\ndef real2() -> int = 2\n"
    (let ((index (agl-imenu-create-index)))
      (should-not (member "hidden" (agl-nav--category-names index "Functions")))
      (should (member "real2" (agl-nav--category-names index "Functions"))))))

(ert-deftest agl-nav-imenu-skips-declaration-inside-triple-quoted-string ()
  (agl-nav--with-buffer
      (concat "let doc = \"\"\"\n"
              "def hidden() -> int = 0\n"
              "\"\"\"\n"
              "def real3() -> int = 3\n")
    (let ((index (agl-imenu-create-index)))
      (should-not (member "hidden" (agl-nav--category-names index "Functions")))
      (should (member "real3" (agl-nav--category-names index "Functions"))))))

;; --- beginning-of-defun return value and motion ---

(ert-deftest agl-nav-beginning-of-defun-at-point-min-returns-nil ()
  (agl-nav--with-buffer "def only() -> int =\n  1\n"
    (goto-char (point-min))
    (should-not (agl-beginning-of-defun))
    (should (= (point) (point-min)))))

(ert-deftest agl-nav-beginning-of-defun-from-nested-scope-declaration ()
  (agl-nav--with-buffer
      (concat "def before() -> int =\n"
              "  1\n"
              "scope Geometry\n"
              "  def area() -> int =\n"
              "    0\n"
              "end Geometry\n")
    (goto-char (agl-nav--pos-after "def area"))
    (beginning-of-defun)
    ;; Defun motion moves over TOP-LEVEL declarations, so the enclosing
    ;; `scope Geometry' header is this position's defun start.
    (should (= (point) (agl-nav--pos-before "scope Geometry")))))

(ert-deftest agl-nav-beginning-of-defun-skips-declaration-inside-triple-quoted-string ()
  (agl-nav--with-buffer
      (concat "let doc = \"\"\"\n"
              "def hidden() -> int = 0\n"
              "\"\"\"\n"
              "def real3() -> int = 3\n")
    (goto-char (agl-nav--pos-after "def hidden() -> int = 0"))
    (beginning-of-defun)
    (should (= (point) (point-min)))))

(ert-deftest agl-nav-beginning-of-defun-from-buffer-end-skips-string-content ()
  (agl-nav--with-buffer
      (concat "let doc = \"\"\"\n"
              "def hidden() -> int = 0\n"
              "\"\"\"\n")
    (goto-char (point-max))
    (beginning-of-defun)
    (should (= (point) (point-min)))))

(ert-deftest agl-nav-beginning-of-defun-returns-non-nil-when-it-moves ()
  (agl-nav--with-buffer "def first() -> int = 0\ndef second() -> int = 0\n"
    (goto-char (point-max))
    (should (agl-beginning-of-defun))
    ;; One call moves to the START OF THE CURRENT declaration, so from
    ;; end-of-buffer that is `def second', not the first declaration.
    (should (= (point) (agl-nav--pos-before "def second")))))

(provide 'agl-navigation-tests)
;;; agl-navigation-tests.el ends here
