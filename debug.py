import numpy as np


E = np.array([[1, 3, 0], [2, 8, 9], [5, 2, 6]])

M = E.copy()
for i in range(1, M.shape[0]):
    for j in range(M.shape[1]):
        if j == 0:
            mini = min(M[i-1, j:j+2])
        elif j == M.shape[1] - 1:
            mini = min(M[i-1, j-1:j+1])
        else:
            mini = min(M[i-1, j - 1:j + 2])
        M[i][j] += mini
print(E)
print(M)


# calculeaza energia dupa ecuatia (1) din articol
"""
        E = compute_energy(img)
        if params.resize_option == "eliminaObiect":
            E[r[1]:r[1]+r[3], r[0]:r[0]+r[2]-i] = -1000
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)
        
        if E[path[-1][0], path[-1][1]] == -1000:
            col = old_col
        else:
            # coloana depinde de coloana pixelului anterior
            if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
                opt = np.argmin(M[line, old_col:old_col + 2])
            elif path[-1][1] == M.shape[1] - 1:  # pixelul este la marginea din dreapta
                opt = np.argmin(M[line, old_col - 1:old_col + 1]) - 1
            else:
                opt = np.argmin(M[line, old_col - 1:old_col + 2]) - 1
            col = path[-1][1] + opt
"""