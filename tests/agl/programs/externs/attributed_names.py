from agl import nominals

Option = nominals.std.option.Option


def first_option(xs):
    return Option.Some(value=xs[0]) if len(xs) else getattr(Option, "None")()


def total(xs):
    return sum(xs)


def words_join(parts, separator):
    return separator.join(parts)
