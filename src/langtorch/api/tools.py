import inspect
import json
import re
from typing import Any, Dict, List


def print_schema(dictionary):
    json_str = json.dumps(dictionary, indent=4, ensure_ascii=False)


def create_json_schema(name, description, parameters, required, hide_details, **additional_details):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    param: {**dict([(k, v) for k, v in details.items() if k not in hide_details]),
                            **additional_details.get(param, {})}
                    for param, details in parameters.items()
                },
                "required": required
            }
        }
    }
    return schema


def parse_docstring(docstring, param_names):
    # Pre-process docstring
    docstring = "\n".join([line.lstrip() for line in docstring.split("\n")]) + "\n"

    # Patterns to detect docstring conventions
    conventions = [
        re.compile(r':param \w+:', re.IGNORECASE),  # Sphinx style
        re.compile(r'Args:', re.IGNORECASE),  # Google style
        re.compile(r'Parameters', re.IGNORECASE)  # Numpydoc style
    ]

    # Detect docstring convention
    convention = None
    description = docstring
    for pattern in conventions:
        match = pattern.search(docstring)
        if match:
            convention = pattern
            description = docstring[:match.start()].strip()
            break

    # Extract parameter info based on the detected convention
    params = {}
    for param in param_names:
        if convention == conventions[0]:  # Sphinx style
            pattern = re.compile(rf':param {param}: (.+?)(?=:param |:return:|$)', re.DOTALL)
        elif convention == conventions[1]:  # Google style
            pattern = re.compile(rf'- {param}(?: \(([^)]+)\))?: (.+?)(?=- \w+(?: \([^)]+\))?: |\n\s*$)', re.DOTALL)
        elif convention == conventions[2]:  # Numpydoc style
            pattern = re.compile(rf'{param}(?: \(([^)]+)\))?: (.+?)(?=\w+(?: \([^)]+\))?: |\n\s*$)', re.DOTALL)
        else:
            # Use a general description pattern as a fallback
            pattern = re.compile(rf'\W+{param}\W+: (.+?)(?=\n\w+\W+:|\w+\W+:|$)', re.DOTALL)

        # Look for matches in the docstring
        matches = pattern.findall(docstring)
        if matches:
            match = matches[0]
            if convention in [conventions[1], conventions[2]]:
                ptype, description_param = match
                param_info = {"description": description_param.strip()}
                if ptype:
                    param_info["type"] = ptype
                params[param] = param_info
            else:
                description_param = match if isinstance(match, str) else match[0]
                params[param] = {"description": description_param.strip()}
        else:
            pattern = re.compile(rf'\W+{param}\W+: (.+?)(?=- \w+ \([^)]+\): |\n\s*$)', re.DOTALL)
            matches = pattern.findall(docstring)
            if matches:
                match = matches[0]
                description_param = match if isinstance(match, str) else match[0]
                params[param] = {"description": description_param.strip()}

    return description, params


def generate_schema_from_function(function: Any, parameter_details: Dict[str, Dict[str, Any]] = None,
                                  hide_details: List[str] = None, include_auto_descriptions=True) -> str:
    if parameter_details is None:
        parameter_details = {}

    if hide_details is None:
        hide_details = []

    # Mapping from Python types to JSON types
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        None: "null"
    }

    # Extract the function name
    function_name = function.__name__

    # Extract the function description and parameter information from the docstring
    docstring = inspect.getdoc(function) or ""
    signature = inspect.signature(function)
    param_names = list(signature.parameters.keys())
    function_description, auto_param_info = parse_docstring(docstring, param_names)

    # Extract the function parameters
    parameters = {}
    required_parameters = []
    for param_name, param in signature.parameters.items():
        param_details = {}

        # Map Python type annotations to JSON data types if not provided in auto_param_info
        if param.annotation is not inspect.Parameter.empty:
            param_details["type"] = type_mapping.get(param.annotation, str(param.annotation))
        elif "type" in auto_param_info.get(param_name, {}).keys():
            param_details["type"] = auto_param_info[param_name].pop("type")

        # Check if the parameter is required (no default value)
        if param.default is not inspect.Parameter.empty:
            param_details["default"] = param.default
        else:
            required_parameters.append(param_name)

        # Include auto extracted descriptions and types
        if include_auto_descriptions and param_name in auto_param_info.keys():
            param_details.update(auto_param_info[param_name])

        # Override with provided parameter_details
        if param_name in parameter_details:
            param_details.update(parameter_details[param_name])
        parameters[param_name] = param_details

    return json.dumps(
        create_json_schema(function_name, function_description, parameters, required_parameters, hide_details,
                           **parameter_details), ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    def revise_your_answer(thought: str, new_answer: str, new_explanation: str):
        """Agent decides to revise their answer based on the other agent's response.

        Parameters:
            thought (str): The internal thought process behind the decision.
            new_answer (str): The revised answer.
            new_explanation (str): The explanation for the revised answer.
        """
        pass


    additional_details = {
        "unit": {
            "enum": ["celsius", "fahrenheit"]
        }
    }

    # json_schema = generate_schema_from_function(revise_your_answer)
