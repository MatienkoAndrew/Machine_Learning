Проект: предсказания победителя в онлайн-игре
---


#### Нужно построить модель, которая по данным о первых пяти минутах матча будет предсказывать его исход — то есть определять команду-победителя.


Dota 2 — многопользовательская компьютерная игра жанра MOBA. Игроки играют между собой матчи. В каждом матче, как правило, участвует 10 человек. Матчи формируются из живой очереди, с учётом уровня игры всех игроков. Перед началом игры игроки автоматически разделяются на две команды по пять человек. Одна команда играет за светлую сторону (The Radiant), другая — за тёмную (The Dire). Цель каждой команды — уничтожить главное здание базы противника, трон.
Вам нужно построить модель, которая по данным о первых пяти минутах матча будет предсказывать его исход — то есть определять команду-победителя.
Чтобы выполнить это задание, вам необходимо провести ряд исследований, сравнить несколько алгоритмов машинного обучения и проверить эффект от ряда манипуляций с признаками. Также, если вам понравится работать с этими данными, вы можете принять участие в соревновании на Kaggle и сравнить свои навыки с другими участниками курса!

### Модель 1: Градиентный бустинг

Один из самых универсальных алгоритмов, изученных в нашем курсе, является градиентный бустинг. Он не очень требователен к данным, восстанавливает нелинейные зависимости, и хорошо работает на многих наборах данных, что и обуславливает его популярность. В данном разделе предлагается попробовать градиентный бустинг для решения нашей задачи.

### Модель 2: Логистическая регрессия

Линейные методы работают гораздо быстрее композиций деревьев, поэтому кажется разумным воспользоваться именно ими для ускорение анализа данных. Одним из наиболее распространенных методов для классификации является логистическая регрессия. В данном разделе предлгается применить ее к данным, а также попробовать различные манипуляции с признаками.