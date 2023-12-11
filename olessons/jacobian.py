from sympy import symbols, diff

def calculate_jacobian(functions, variables):
    """
    

    Parameters:
    functions (list):
    variables (list): 

    Returns:
    list of lists: 
    """
    jacobian_matrix = []

    for func in functions:
        row = [diff(func, var) for var in variables]
        jacobian_matrix.append(row)

    return jacobian_matrix

x, y, z = symbols('x y z')
f1 = x**2 + y**2
f2 = x * y * z
f3 = x + y + z

functions_list = [f1, f2, f3]
variables_list = [x, y, z]

jacobian_matrix_result = calculate_jacobian(functions_list, variables_list)
print('s')
for row in jacobian_matrix_result:
    print(row)