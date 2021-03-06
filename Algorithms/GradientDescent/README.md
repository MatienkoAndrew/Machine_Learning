Градиентный спуск
---


***Градиентный спуск*** представляет собой самый общий алгоритм оптимизации, способный находить оптимальные решения широкого диапазона задач. Основная идея градиентного спуска заключается в том, чтобы итеративно подстраивать параметры для сведения к минимуму функции издержек


![Безымянный](https://user-images.githubusercontent.com/29499863/78986290-7590a000-7b1a-11ea-9c34-a6a46ab18c8a.png)

> Необходимо масштабирование данных

![Безымянный](https://user-images.githubusercontent.com/29499863/78986346-a07af400-7b1a-11ea-8ffa-fece3be8d094.png)


# Пакетный градиентный спуск

![Безымянный](https://user-images.githubusercontent.com/29499863/78986420-bdafc280-7b1a-11ea-8ea3-8d2962a83b9f.png)


![Безымянный](https://user-images.githubusercontent.com/29499863/78986444-cef8cf00-7b1a-11ea-89a9-8fe9432a37c6.png)

![Безымянный](https://user-images.githubusercontent.com/29499863/78986479-e1730880-7b1a-11ea-8b3d-d01165e1af9f.png)

![Безымянный](https://user-images.githubusercontent.com/29499863/78986516-f780c900-7b1a-11ea-8538-aa0bff017f23.png)

![Безымянный](https://user-images.githubusercontent.com/29499863/78986554-0d8e8980-7b1b-11ea-8c7a-21c288320058.png)


# Стохастический градиентный спуск

Cтохастический градиентный спуск (Stochastic Gradient Descent - SGD) на каждом шаге просто выбирает из обучающего набора случайный образец и вычисляет градиенты на основе только этого единственного образца. Очевидно, алгоритм становится гораздо быстрее, т.к. на каждой операции ему приходится манипулировать совсем малым объемом данных. Также появляется возможность проводить обучение на гигантских обучающих наборах, потому что на каждой итерации в памяти должен находиться только один образец 

![Безымянный](https://user-images.githubusercontent.com/29499863/78986656-47f82680-7b1b-11ea-8e45-518e399c6030.png)


# Мини-пакетный градиентный спуск

На каждом шаге вместо вычисления градиентов на основе полного обучающего набора (как в пакетном градиентном спуске) или только одного образца (как в стохастическом градиентном спуске) мини-пакетный градиентный спуск вычисляет градиенты на небольших случайных наборах образцов, которые называются мини-пакетами (*mini-batch*). 

# Вывод

![Безымянный](https://user-images.githubusercontent.com/29499863/78986777-9d343800-7b1b-11ea-8596-edd0092a72f5.png)
