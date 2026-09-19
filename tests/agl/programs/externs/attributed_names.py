from agl import option_none, option_some


def first_option(xs):
    return option_some(xs[0]) if len(xs) else option_none()


def total(xs):
    return sum(xs)


def words_join(parts, separator):
    return separator.join(parts)
