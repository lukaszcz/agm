import sys

# AgL ints are unbounded, so every AGM process lifts Python's int/str digit limit on
# first import. Accepted trade-off: huge numbers are admitted; conversion is subquadratic,
# and its cost scales with input size.
sys.set_int_max_str_digits(0)
