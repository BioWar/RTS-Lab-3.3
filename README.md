## RTS Lab 3.3: Genetic algorithm for Diophantine equation (Android/Python)
### IO-71 Vorotyntsev Petro

__Task__: Make genetic algorithm implementation for Android device with opportunity to enter data and find solution of Diophantine equation.

__Screenshot of program__:

![alt text](https://github.com/BioWar/RTS-Lab-3.3/blob/master/Screenshots/rts_lab_33_1.jpg)

__Additional task__: нехай для однієї і тієї ж задачі збільшується % мутації, визначити оптимальний % для поточної задачі

__Implementation__: для того, щоб знайти найкращі параметри для мутації я використав щось на зразок Grid Search алгоритму, який перевіряє комбінації параметрів та прериває цикл якщо суттевих різниці між Y - Y_predicted < 0.1. В результаті я вивожу найкращу кількість параметрів для мутації та діапазон мутації.

Тестування показало що при вдалій ініціалізації та високій мутації кількість ітерацій для сходження алгоритма може досягати 2. Але зазвичай при максимальній кількості ітерацій рівній 30, алгоритм сходиться на 15-25 ітерації.

__Screenshot of additional task output__:
![alt text](https://github.com/BioWar/RTS-Lab-3.3/blob/master/Screenshots/grid_search_params_rts_lab33_2.png)

