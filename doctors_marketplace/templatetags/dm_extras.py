from django import template

register = template.Library()

@register.filter
def split(value, sep=','):
    if not value:
        return []
    return [s.strip() for s in str(value).split(sep) if s.strip()]
