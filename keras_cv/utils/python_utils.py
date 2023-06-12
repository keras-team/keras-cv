# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities with miscellaneous python extensions."""


class classproperty(property):
    """Define a class level property."""

    def __get__(self, _, owner_cls):
        return self.fget(owner_cls)


def format_docstring(**replacements):
    """Format a python docstring using a dictionary of replacements.

    This decorator can be placed on a function, class or method to format it's
    docstring with python variables.

    The decorator will replace any double bracketed variable with a kwargs
    value passed to the decorator itself. For example
    `@format_docstring(name="foo")` will replace any occurrence of `{{name}}` in
    the docstring with the string literal `foo`.
    """

    def decorate(obj):
        doc = obj.__doc__
        # We use `str.format()` to replace variables in the docstring, but use
        # double brackets, e.g. {{var}}, to mark format strings. So we need to
        # to swap all double and single brackets in the source docstring.
        doc = "{".join(part.replace("{", "{{") for part in doc.split("{{"))
        doc = "}".join(part.replace("}", "}}") for part in doc.split("}}"))
        obj.__doc__ = doc.format(**replacements)
        return obj

    return decorate
