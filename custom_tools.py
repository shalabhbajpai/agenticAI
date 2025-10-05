from langchain_community.tools import tool

# Step 1  -  create a function

#def add(a, b):
#    ''' Adding two numbers '''
 #   return a + b

# Step 2  - add type hints

#def add(a :int, b:int) -> int :
#    ''' Adding two numbers '''
#    return a + b 

# Step 3  - add tool decorator

@tool
def add(a :int, b:int) -> int :
    ''' Adding two numbers '''
    return a + b 


@tool
def multiply(a :int, b:int) -> int :
    ''' Adding two numbers '''
    return a * b 





result = add.invoke({"a":10,"b":5})

print(result)
print(add.name)
print(add.description)
print(add.args)


class MathToolkit:
    def get_tools(self):
        return [add, multiply]
    


toolkit= MathToolkit()

tools = toolkit.get_tools()

for tool in tools:
    print(tool.name +" -> "+ tool.description)
    