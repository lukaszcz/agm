import sys

# AgL ints are unbounded; Python's int/str digit limit is an obsolete DoS guard
# (conversion is subquadratic), so every AGM process lifts it on first import.
sys.set_int_max_str_digits(0)
