// Tests for the AgL micro syntax rules in ../agl.yaml.
//
// The rules are driven through micro's own highlighter rather than a
// reimplementation of it: the package below is the one the editor runs, so
// these tests exercise the real regexp engine (Go RE2, no lookaround), the
// real rule precedence (a later rule repaints an earlier one) and the real
// region handling for strings and comments. A model of that behaviour written
// here would only ever test the model.
package aglmicro

import (
	"os"
	"strings"
	"testing"

	"github.com/zyedidia/micro/v2/pkg/highlight"
)

const rulesPath = "../agl.yaml"

// faces returns the highlight group covering each rune of source, one entry
// per rune, with "" meaning no group.
func faces(t *testing.T, source string) []string {
	t.Helper()
	rules, err := os.ReadFile(rulesPath)
	if err != nil {
		t.Fatalf("reading %s: %v", rulesPath, err)
	}
	file, err := highlight.ParseFile(rules)
	if err != nil {
		t.Fatalf("parsing %s: %v", rulesPath, err)
	}
	header, err := highlight.MakeHeaderYaml(rules)
	if err != nil {
		t.Fatalf("reading the header of %s: %v", rulesPath, err)
	}
	def, err := highlight.ParseDef(file, header)
	if err != nil {
		t.Fatalf("compiling %s: %v", rulesPath, err)
	}

	matches := highlight.NewHighlighter(def).HighlightString(source)
	var out []string
	for i, line := range strings.Split(source, "\n") {
		// A LineMatch maps a column to the group that starts there and runs
		// until the next entry, so the groups have to be carried forward.
		group := ""
		for col := range []rune(line) {
			if i < len(matches) {
				if g, ok := matches[i][col]; ok {
					group = g.String()
				}
			}
			out = append(out, group)
		}
		out = append(out, "") // the newline itself carries no group
	}
	return out
}

// assertFace fails unless every rune of substring in source carries group.
// substring must occur exactly once, so that a test cannot silently start
// checking a different occurrence than it was written for.
func assertFace(t *testing.T, source, substring, group string) {
	t.Helper()
	first := strings.Index(source, substring)
	if first < 0 {
		t.Fatalf("%q does not occur in %q", substring, source)
	}
	if strings.Index(source[first+1:], substring) >= 0 {
		t.Fatalf("%q occurs more than once in %q", substring, source)
	}
	all := faces(t, source)
	start := len([]rune(source[:first]))
	// Report the substring's whole face at once: a rule that faces it wrongly
	// usually does so for every rune, and one line per rune buries the cause.
	var got []string
	for i := range []rune(substring) {
		face := all[start+i]
		if face == "" {
			face = "-"
		}
		if len(got) == 0 || got[len(got)-1] != face {
			got = append(got, face)
		}
	}
	if len(got) != 1 || got[0] != group {
		t.Errorf("in %q: %q faced as %s, want %q",
			source, substring, strings.Join(got, "+"), group)
	}
}

// The builtin call names, mirroring BUILTIN_CALL_NAMES in
// src/agm/agl/scope/symbols.py. Keep this list in step with that one.
var builtins = []string{
	"print", "render", "exec", "ask", "ask-request",
	"copy", "shallow_copy", "resource", "resource-dir",
}

func TestBuiltinsAreFacedBySpelling(t *testing.T) {
	for _, name := range builtins {
		assertFace(t, "let x = "+name+"(a)", name, "identifier")
	}
}

// A raw-tail opener starts the verbatim payload region below, so it faces with
// the payload rather than as a builtin call: micro paints a region's start
// delimiter with the region's own group. Mirrors RAW_TAIL_NAMES in
// src/agm/raw_tail_catalog.py.
func TestRawTailOpenersAreFaced(t *testing.T) {
	assertFace(t, "exec$ ls -la", "exec$", "constant.string")
	assertFace(t, "ask$ summarise this", "ask$", "constant.string")
	// The bare builtin spellings, without the `$', are unaffected.
	assertFace(t, "let x = exec(c)", "exec", "identifier")
	assertFace(t, "let x = ask(p)", "ask", "identifier")
}

// A raw-tail opener that merely ends a longer name is part of that name, so
// it must not open the verbatim region: `-' continues an AgL identifier
// (IDENT_STOP in src/agm/util/ident.py) even though `\b' sees a boundary
// there. A region's start delimiter cannot be repainted by a later rule, so
// the boundary has to be in the delimiter itself.
func TestRawTailOpenerInsideANameStaysCode(t *testing.T) {
	assertFace(t, "let x = do-exec$ + 1", "do-exec$", "default")
	assertFace(t, "let x = do-exec$ + 1", "+", "symbol.operator")
	assertFace(t, "let y = an-ask$ + 1", "an-ask$", "default")
	// A spelling inside a string literal is string content, not an opener.
	assertFace(t, `print("exec$ ls")`, "print", "identifier")
}

// `-', `?', `!' and `$' continue an AgL name (IDENT_STOP in
// src/agm/util/ident.py), so a keyword or builtin spelling that merely starts
// or ends a longer name must not be faced. Regexp `\b' does not know that;
// these pin the layering in agl.yaml that compensates for it.
func TestLongerIdentifiersStayPlain(t *testing.T) {
	for _, name := range []string{
		"resource-path",     // starts with the builtin `resource'
		"copy-of",           // starts with the builtin `copy', ends with the keyword `of'
		"ask-request-later", // starts with the whole builtin `ask-request'
		"x-resource",        // ends with a builtin
		"record-id",         // starts with a keyword
		"var-limit",         // starts with the field-marker keyword
		"and-then",          // starts with a keyword
		"not-found",         // starts with a keyword
		"do-it!",            // starts with a keyword, ends in `!'
		"is-ready?",         // starts with a keyword, ends in `?'
		"x-1",               // ends in a digit run that is not a literal
	} {
		assertFace(t, "let "+name+" = 1", name, "default")
	}
}

func TestHyphenatedBuiltinsFaceAsOneName(t *testing.T) {
	// The `-' is part of the name, not an operator between two names.
	assertFace(t, "let x = ask-request", "ask-request", "identifier")
	assertFace(t, "let x = resource-dir", "resource-dir", "identifier")
}

func TestKeywordsAreFaced(t *testing.T) {
	assertFace(t, "let x = 1", "let", "statement")
	assertFace(t, "for i in 1 to 10 do", "for", "statement")
	assertFace(t, "for i in 1 to 10 do", "do", "statement")
	assertFace(t, "import std/text", "import", "preproc")
	assertFace(t, "let w = x as? int", "as?", "statement")
}

func TestTypesAndLiterals(t *testing.T) {
	assertFace(t, "def f(a: int) -> text", "int", "type")
	assertFace(t, "let b = true", "true", "constant.bool")
	assertFace(t, "let n = null", "null", "constant")
	// The `.' of a decimal literal must not be repainted as punctuation.
	assertFace(t, "let d = 3.14", "3.14", "constant.number")
}

func TestStringsAndComments(t *testing.T) {
	assertFace(t, `let s = "plain"`, `"plain"`, "constant.string")
	assertFace(t, "# a remark", "# a remark", "comment")
	// Keywords inside a string or a comment are text, not keywords.
	assertFace(t, `let s = "let"`, `"let"`, "constant.string")
	assertFace(t, "# let x", "let x", "comment")
	// Interpolation delimiters stay visible inside a template.
	assertFace(t, `let s = "hi %{name}"`, "%{", "special")
}

// A `var' marker declares a mutable record or enum-member field, so the
// keyword has to face wherever a field is declared -- in a layout body, in an
// inline field list, and in an enum member's payload.
func TestMutableFieldMarkerIsFaced(t *testing.T) {
	assertFace(t, "record Counter(var value: int)", "var", "statement")
	assertFace(t, "record Counter\n  var value: int", "var", "statement")
	assertFace(t, "enum Cell\n  Full(var payload: text)", "var", "statement")
	// The declared type still faces as a type through the marker.
	assertFace(t, "record Counter(var value: int)", "int", "type")
}

// `::' and `:=' are single tokens whose `:' would otherwise be repainted as a
// plain delimiter, splitting the token across two faces.
func TestColonOperatorsFaceAsOneToken(t *testing.T) {
	assertFace(t, "count := 1", ":=", "symbol.operator")
	assertFace(t, "meter.value := 7", ":=", "symbol.operator")
	assertFace(t, "let o = Point::origin", "::", "symbol.operator")
	// A lone `:' introducing an annotation stays a plain delimiter.
	assertFace(t, "def f(a: int)", ":", "symbol")
}

// A symbolic operator name is one token, however many characters it spells.
// `std/core' declares `|>', `<|', `>>' and `<<', and a program may declare any
// other spelling with `infixl'/`infixr'.
func TestOperatorNamesFaceAsOneToken(t *testing.T) {
	for _, op := range []string{"|>", "<|", ">>", "<<", "++", "<$>"} {
		assertFace(t, "let y = a "+op+" b", op, "symbol.operator")
	}
	assertFace(t, "infixl |> at 5", "|>", "symbol.operator")
	// A lone `|' is the branch marker, a plain delimiter rather than an operator.
	assertFace(t, "case v of\n| A => 1", "|", "symbol")
}

// `?' and `?N' are partial-application placeholders: whole tokens, so the
// digits of `?1' are part of the placeholder rather than a number literal.
func TestPlaceholdersFaceAsOneToken(t *testing.T) {
	assertFace(t, "let g = f(?)", "?", "symbol.operator")
	assertFace(t, "let g = f(?1, ?2)", "?1", "symbol.operator")
	assertFace(t, "let g = f(?1, ?2)", "?2", "symbol.operator")
	// The `?' ending a name is part of that name, not a placeholder.
	assertFace(t, "let u = is-ready?", "is-ready?", "default")
}

// A re-faced builtin spelling has to consume the delimiter that follows it,
// which then needs its own face back.
func TestDelimiterAfterARefacedNameIsRepainted(t *testing.T) {
	assertFace(t, "f(ask-request, x)", ",", "symbol")
	assertFace(t, "f(resource-dir, x)", ",", "symbol")
}

// A raw-tail payload is verbatim text, so AgL spellings inside it are not
// code: the whole tail faces as a string, opener included.
func TestRawTailPayloadIsVerbatim(t *testing.T) {
	assertFace(t, "let a = exec$ ls -la | grep record", "exec$ ls -la | grep record", "constant.string")
	assertFace(t, "let a = ask$ summarise the record", "ask$ summarise the record", "constant.string")
	// The code before the opener is unaffected.
	assertFace(t, "let a = exec$ ls", "let", "statement")
}
