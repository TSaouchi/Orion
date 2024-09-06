# Path and Char manipulation
import os

# I/O
import ast

# Orion
import Core as Orion

class Parser:
    """
    Parser class for parsing and extracting information from a Python script.

    :param script_path: The path to the Python script to be parsed.
    :type script_path: str

    :ivar script_path: The path to the Python script.
    :ivar tree: The abstract syntax tree (AST) representation of the parsed script.
    :vartype script_path: str
    :vartype tree: ast.Module

    Methods:
    :meth:`_parse_script`: Parse the script and generate the abstract syntax tree (AST).
    :meth:`get_dictionary`: Extract variable assignments from the parsed script.
    :meth:`extract_set`: Extract information from set methods in the parsed script.
    """
    def __init__(self, script_path):
        """
        Initialize the Parser with the path to the script.

        :param script_path: The path to the Python script to be parsed.
        :type script_path: str
        """
        self.script_path = script_path
        self.tree = self._parse_script()


    def get_dictionary(self, field_to_read):
        """
        Extract variable assignments from the parsed script.

        :param field_to_read: The name of the variable to retrieve.
        :type field_to_read: str

        :return: The value of the specified variable, or None if not found.
        :rtype: Any
        """
        parsed_code = self.tree

        variables = {}
        for node in parsed_code.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variable_name = target.id
                        variable_value = eval(compile(ast.Expression(node.value), '<string>', 'eval'))
                        variables[variable_name] = variable_value

        return variables.get(field_to_read)


    def extract_set(self, set_method):
        """
        Convert a list of ASCII values to a string of characters.

        :param ascii_values: List of ASCII values representing characters.
        :type ascii_values: list[int]

        :return: The resulting string obtained by converting the ASCII values.
        :rtype: str
        """
        set_dict = {}
        object_name, subobject_method = set_method.split('.')
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == object_name
                and node.value.func.attr == subobject_method
                and len(node.value.args) == 2
                and isinstance(node.value.args[0], ast.Str)
            ):
                name = eval(compile(ast.Expression(node.value.args[0]), '<string>', 'eval'))
                value = eval(compile(ast.Expression(node.value.args[1]), '<string>', 'eval'))  #:Use .s for string values
                set_dict[name] = value

        return set_dict

    @classmethod
    def ascii_values_to_text(cls, ascii_values):
        """
        Convert a list of ASCII values to a string of characters.
        """
        translated_text = ''.join(chr(value) for value in ascii_values)
        return translated_text


    def _parse_script(self):
        """
        Parse the script and generate the abstract syntax tree (AST).

        :return: The AST representation of the parsed script.
        :rtype: ast.Module
        """
        with open(self.script_path, 'r') as file:
            script_content = file.read()
        return ast.parse(script_content, filename=self.script_path)