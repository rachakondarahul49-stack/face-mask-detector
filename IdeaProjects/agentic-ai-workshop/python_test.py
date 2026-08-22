from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

result = python_repl.run("print(987654 * 123456)")

print("Result:", result)