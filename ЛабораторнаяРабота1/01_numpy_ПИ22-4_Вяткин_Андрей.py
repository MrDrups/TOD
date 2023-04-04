import numpy as np


print("Задание 1")
data = np.loadtxt("./data/minutes_n_ingredients.csv", dtype=int, delimiter=",", skiprows=1)
print(data[:5], end='\n')


print("Задание 2")
for column in range(1, 3):
    print(f"Минимальное: {np.min(data[:, column])}", f"Максимальное: {np.max(data[:, column])}",\
    f'Среднее: {np.mean(data[:, column])}', f"Медиана: {np.median(data[:, column])}", end='\n')


print("3. Ограничьте сверху значения продолжительности выполнения рецепта значением квантиля $q_{0.75}$. ")
time_limit = data.copy()
limit = np.quantile(time_limit[:, 1], q=0.75, axis=0)
time_limit[:, 1][time_limit[:, 1] > limit] = limit
print(time_limit, end='\n')


print("4. Посчитайте, для скольких рецептов указана продолжительность, равная нулю. Замените для таких строк значение в данном столбце на 1.")
notzero = time_limit.copy()
mask = notzero[:, 1] == 0
zeros_n = np.count_nonzero(mask)
notzero[:, 1][mask] = 1
print(zeros_n)
print(notzero, end='\n')


print("5. Посчитайте, сколько уникальных рецептов находится в датасете.")
print(np.unique(notzero[:, 1:], axis=0).shape[0], end='\n')


print("6. Сколько и каких различных значений кол-ва ингредиентов присутвует в рецептах из датасета?")
print(len(np.unique(notzero[:, 2])), end='\n')


print("7. Создайте версию массива, содержащую информацию только о рецептах, состоящих не более чем из 5 ингредиентов.")
print(notzero[notzero[:, 2] <= 5], end='\n')


print("8. Для каждого рецепта посчитайте, сколько в среднем ингредиентов приходится на одну минуту рецепта.\n "
      "Найдите максимальное значение этой величины для всего датасета")
ingredients = notzero.copy()
print((ingredients[:, 2]/ingredients[:, 1]).max(), end='\n')


print("9. Вычислите среднее количество ингредиентов для топ-100 рецептов с наибольшей продолжительностью")
top_100 = np.argsort(notzero[:, 1])[:-100:-1]
print(notzero[top_100][:, 2].mean(), end='\n')


print("10. Выберите случайным образом и выведите информацию о 10 различных рецептах")
print(data[np.random.randint(data.shape[0], size=10), :], end='\n')


print("11. Выведите процент рецептов, кол-во ингредиентов в которых меньше среднего.")
mask = data[:, 2] < notzero[top_100][:, 2].mean()
print(len(data[mask])/len(data) * 100, end='\n')


print("12. Назовем (простым) такой рецепт, длительность выполнения которого не больше 20 минут и кол-во ингредиентов в котором не больше 5.\n"
      " Создайте версию датасета с дополнительным столбцом, значениями которого являются 1, если рецепт простой, и 0 в противном случае.")
added_column = data.copy()
mask = (data[:, 1] <= 20) & (data[:, 2] <= 5)
column = mask.astype(int)
added_column = np.column_stack([added_column, column])
print(added_column, end='\n')


print("13. Выведите процент (простых) рецептов в датасете")
mask = added_column[:, 3] == 1
temp = added_column[:, 3][mask]
print(len(temp)/len(added_column) * 100, end='\n')


print("14. Разделим рецепты на группы по следующему правилу. Назовем рецепты короткими, если их продолжительность составляет менее 10 минут;\n"
      " стандартными, если их продолжительность составляет более 10, но менее 20 минут;\n"
      " и длинными, если их продолжительность составляет не менее 20 минут. \n"
      "Создайте трехмерный массив, где нулевая ось отвечает за номер группы (короткий, стандартный или длинный рецепт), \n"
      "первая ось - за сам рецепт и вторая ось - за характеристики рецепта. \n"
      "Выберите максимальное количество рецептов из каждой группы таким образом, чтобы было возможно сформировать трехмерный массив.\n"
      " Выведите форму полученного массива.")
fast_rec = data[data[:, 1] < 10]
standard_arr = data[(data[:, 1] >= 10) & (data[:, 1] < 20)]
long_arr = data[20 <= data[:, 1]]
crop_value = np.min([fast_rec.shape[0], standard_arr.shape[0], long_arr.shape[0]])
result_arr = np.array([fast_rec[:crop_value], standard_arr[:crop_value], long_arr[:crop_value]])
print(result_arr, end='\n')

