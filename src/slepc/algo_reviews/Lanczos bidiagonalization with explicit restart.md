## Введение ##
Этот отчёт не включает материалы о несимметричной версии метода Ланцоша. Собственные значения, основанные на этом методе, могут быть добавлены в будущие версии slepc.
Важные определения:
<p>Подпространство Крылова - линейное пространство, порождённое вектором v &in; C <sup>n</sup> и матрицей A &in;
C<sup>n*n</sup> вида : K<sub>m</sub>(v,A) = span{v,Av,A<sup>2</sup>v,...,A<sup>m-1</sup>v}.</p>

## Описание алгоритма ##
Этот раздел предоставляет обзор метода Ланцоша и некоторых его вариаций, включая методы для предотвращения потери ортогональности.
Метод Ланцоша может быть выведен с разных точек зрения. Один из таких подходов заключается в приведении симметричной матрицы <i>A</i> порядка <i>n × n</i> к тридиагональной форме с помощью трёхчленной рекуррентной формулы. Задав начальный вектор <i>v<sub>1</sub></i> с единичной нормой и положив <i>β<sub>1</sub> = 0</i>, применяют следующую рекуррентную формулу:

<p align="center">β<sub>j+1</sub>v<sub>j+1</sub> = Av<sub>j</sub> − α<sub>j</sub> v<sub>j</sub> − β<sub>j</sub> v<sub>j−1</sub>(1),</p>

где <i>α<sub>j</sub> = v<sub>j</sub><sup>*</sup> A v<sub>j</sub></i> и <i>β<sub>j+1</sub> = v<sub>j+1</sub><sup>*</sup> A v<sub>j</sub></i>, что порождает ортонормированный набор векторов Ланцоша <i>v<sub>j</sub></i> и тридиагональную матрицу, определённую как

<table border="1" cellpadding="5" cellspacing="0">
  <tr>
    <td>α<sub>1</sub></td>
    <td>β<sub>2</sub></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>β<sub>2</sub></td>
    <td>α<sub>2</sub></td>
    <td>β<sub>3</sub></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>β<sub>3</sub></td>
    <td>α<sub>3</sub></td>
    <td>⋱</td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>⋱</td>
    <td>⋱</td>
    <td>β<sub>n</sub></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td>β<sub>n</sub></td>
    <td>α<sub>n</sub></td>
  </tr>
</table>


Можно показать, что вектор <i>v<sub>n+1</sub></i> равен нулю, и выполняется следующее соотношение:
<p align="center">A V − V T = 0 (3),</p>

где <i>V = [v<sub>1</sub>, v<sub>2</sub>, . . . , v<sub>n</sub>]</i>. То есть рекурсия Ланцоша вычисляет тридиагональную матрицу, которая ортогонально подобна <i>A</i>.

При описании алгоритма вычисления рекурсии Ланцоша можно учитывать два замечания. Первое заключается в том, что <i>β<sub>j+1</sub></i> можно вычислить как <i>‖Av<sub>j</sub> − α<sub>j</sub> v<sub>j</sub> − β<sub>j</sub> v<sub>j−1</sub>‖<sub>2</sub></i>, поскольку <i>v<sub>j+1</sub></i> имеет единичную норму. Численный анализ, проведенный Пейджем [1972], показывает, что этот альтернативный подход улучшает численную устойчивость при реализации рекурсии в арифметике с конечной точностью. 

Второе замечание заключается в том, что <i>α<sub>j</sub></i> можно вычислить как <i>v<sub>j</sub><sup>*</sup>(Av<sub>j</sub> − β<sub>j</sub> v<sub>j−1</sub>)</i>, поскольку <i>v<sub>j</sub></i> и <i>v<sub>j−1</sub></i> ортогональны по конструкции. Этот альтернативный подход также поддерживается Пейджем [1980]. Учитывая эти замечания, основной алгоритм Ланцоша можно записать как в Алгоритме 1.

### Алгоритм 1 (Базовый алгоритм Ланцоша — вид рекурсии) ###
(Lanczos Methods in SLEPc страница 3)

**Входные данные:** Матрица <i>A</i>, число шагов <i>m</i> и начальный вектор <i>v<sub>1</sub></i> нормы 1.  
**Выходные данные:** (<i>V<sub>m</sub>, T<sub>m</sub>, v<sub>m+1</sub>, β<sub>m+1</sub></i>) так, что 
1. Выберите вектор единичной нормы <i>v<sub>1</sub></i>.
2. Установите <i>β<sub>1</sub> = 0</i>.
3. Для <i>j = 1, 2, . . .</i>:
   - <i>u<sub>j+1</sub> = Av<sub>j</sub> − β<sub>j</sub> v<sub>j−1</sub></i>
   - <i>α<sub>j</sub> = v<sub>j</sub><sup>*</sup> u<sub>j+1</sub></i>
   - <i>u<sub>j+1</sub> = u<sub>j+1</sub> − α<sub>j</sub> v<sub>j</sub></i>
   - <i>β<sub>j+1</sub> = ‖u<sub>j+1</sub>‖<sub>2</sub></i>
   - Если <i>β<sub>j+1</sub> = 0</i>, остановитесь.
   - <i>v<sub>j+1</sub> = u<sub>j+1</sub>/β<sub>j+1</sub></i>.
4. Конец.

Где: <p>v<sub>1</sub> - начальный вектор, который должен быть единичной нормы. Этот вектор служит начальной точкой для построения ортогональной базы пространства Крылова.</p>
<p>&alpha;,&beta; - скалярные коэффициенты тридиагональной матрицы.</p>
<p>v<sub>j</sub> - базисный ортогонализованный вектор, принадлежащий Крыловскому подпространству. </p>
<p>u<sub>j</sub> - промежуточный вектор для вычесления v<sub>j</sub>.</p>

Конечно, алгоритм Ланцоша наиболее полезен, когда не вычисляются все <i>n</i> векторы, что невозможно в контексте очень больших матриц. Если выполняется только <i>m</i> шагов Ланцоша, то вместо уравнения (3) имеет место следующее соотношение:

<p align="center">A V<sub>m</sub> − V<sub>m</sub>T<sub>m</sub> = β<sub>m+1</sub> v<sub>m+1</sub> e<sub>m</sub><sup>*</sup>(4),</p>

где <i>T<sub>m</sub></i> — это ведущая подматрица размера <i>m × m</i> матрицы <i>T</i>, а <i>V<sub>m</sub> = [v<sub>1</sub>, v<sub>2</sub>, . . . , v<sub>m</sub>]</i>. Уравнение (4) описывает остаток <i>m</i>-шаговой факторизации Арнольди. То есть процесс Ланцоша также можно рассматривать как вычисление ортогонального проекции матрицы <i>A</i> на Крыловское подпространство <i>K<sub>m</sub>(A, v<sub>1</sub>)</i>. С этой точки зрения метод Ланцоша эквивалентен методу Арнольди, см. Алгоритм 2.

### Алгоритм 2 (Базовый алгоритм Ланцоша — вид проекции) 
(Lanczos Methods in SLEPc страница 4)
**Входные данные:** Матрица <i>A</i>, число шагов <i>m</i> и начальный вектор <i>v<sub>1</sub></i> нормы 1.  
**Выходные данные:** (<i>V<sub>m</sub>, T<sub>m</sub>, v<sub>m+1</sub>, β<sub>m+1</sub></i>) так, что 

<p align="center">A V<sub>m</sub> − V<sub>m</sub>T<sub>m</sub> = β<sub>m+1</sub> v<sub>m+1</sub> e<sub>m</sub><sup>*</sup></p>

Для <i>j = 1, 2, . . . , m</i>:
- <i>u<sub>j+1</sub> = A v<sub>j</sub></i>
- Ортогонализируйте <i>u<sub>j+1</sub></i> относительно <i>V<sub>j</sub></i> (вычисляя <i>α<sub>j</sub></i>)
- <i>β<sub>j+1</sub> = ‖u<sub>j+1</sub>‖<sub>2</sub></i>
- Если <i>β<sub>j+1</sub> = 0</i>, остановитесь
- <i>v<sub>j+1</sub> = u<sub>j+1</sub>/β<sub>j+1</sub></i>

Конец.

Где:
<p> Матрица A - исходная матрица.</p>
<p> Число шагов m - количество итераций алгоритма. Определяет размер Крыловского подпространства, которое будет построено и размер итоговой трёхдиагональной матрицы T<sub>m</sub>.</p>
<p> v1<sub>1</sub> - начальный вектор с которого начинается построение Крыловского подпространства, обычно он нормирован, т.е. ||v<sub>1</sub>|| = 1.</p>
<p> V<sub>m</sub> - матрица, составленная из векторов v<sub>1</sub>,v<sub>2</sub>...v<sub>m</sub>,  которые образуют ортонормированный базис для подпространства K<sub>m</sub>(A,v<sub>1</sub>)</p>
<p> T<sub>m</sub> - трёхдиагональная матрица размером m*m, которая является приближенной версией матрицы А в базисе V<sub>m</sub></p>
<p>v<sub>m+1</sub> - вектор полученный на последнем шаге алгоритма, нужен для анализа ошибки приближения</p>
<p>&beta;<sub>m+1</sub> - норма вектора v<sub>m+1</sub>, которая показывает насколько далеко следующий шак уходит за пределы подпространства, если равен нулю, это значит, что подпространство не может быть дальше расширенно.</p>
В Алгоритме 2 вторая строка в цикле выполняет процесс Грама-Шмидта для ортогонализации вектора <i>u<sub>j+1</sub></i> относительно столбцов <i>V<sub>j</sub></i>, то есть векторов <i>v<sub>1</sub>, v<sub>2</sub>, . . . , v<sub>j</sub></i> (см. технический отчет SLEPc STR-1, «Рутины ортогонализации в SLEPc» для подробностей о Граме-Шмидте). В этой операции вычисляются <i>j</i> коэффициентов Фурье. В точной арифметике первые <i>j−2</i> коэффициента равны нулю, и, следовательно, соответствующие операции не нужно выполнять (ортогональность относительно первых <i>j − 2</i> векторов является автоматической). Другие два коэффициента — это <i>β<sub>j</sub></i> и <i>α<sub>j</sub></i>. Обратите внимание, что, согласно Пейджу, <i>β<sub>j</sub></i>, вычисленный в этой операции, следует отбросить, и вместо него использовать значение <i>‖u<sub>j</sub>‖<sub>2</sub></i>, вычисленное на предыдущей итерации. С точки зрения ортогонализации, Алгоритм 1 выполняет модифицированный шаг Грама-Шмидта только с векторами <i>v<sub>j−1</sub></i> и <i>v<sub>j</sub></i>, в то время как вычисление <i>α<sub>j</sub></i> как <i>v<sub>j</sub><sup>*</sup>Av<sub>j</sub></i> соответствовало бы классическому Граму-Шмидту.

В дальнейшем мы сосредоточим наше обсуждение на Алгоритме 2, поскольку ортогонализация будет ключевым аспектом устойчивых вариантов Ланцоша, которые справляются с потерей ортогональности.

Как и в случае с методом Арнольди, поскольку <i>V<sub>m</sub><sup>*</sup>v<sub>m+1</sub> = 0</i> по конструкции, то, предварительно умножив уравнение (4) на <i>V<sub>m</sub><sup>*</sup></i>, получаем:

<p align="center">V<sub>m</sub><sup>*</sup>A V<sub>m</sub> = T<sub>m</sub>(5),</p>
То есть матрица <i>T<sub>m</sub></i> представляет собой ортогональную проекцию матрицы <i>A</i> на криловское подпространство, и этот факт позволяет нам вычислять приближения Рейли-Ритца для собственных пар матрицы <i>A</i>. Пусть <i>(λ<sub>i</sub>, y<sub>i</sub>)</i> — собственная пара матрицы <i>T<sub>m</sub></i>, тогда значение Ритца <i>λ<sub>i</sub></i> и вектор Ритца <i>x<sub>i</sub> = V<sub>m</sub>y<sub>i</sub></i> могут быть приняты в качестве приближений для собственных пар матрицы <i>A</i>. Обычно лишь небольшая часть из <i>m</i> приближений является качественной. Это можно оценить с помощью нормы остатка для пары Ритца, которую довольно легко вычислить:

<p align="center">‖A x<sub>i</sub> − λ<sub>i</sub>x<sub>i</sub}‖<sub>2</sub> = ‖A V<sub>m</sub>y<sub>i</sub> − λ<sub>i</sub>V<sub>m</sub>y<sub>i</sub>‖<sub>2</sub> = ‖(AV<sub>m</sub> − V<sub>m</sub>T<sub>m</sub>)y<sub>i</sub>‖<sub>2</sub> = β<sub>m+1</sub>|e<sub>m</sub><sup>*</sup>y<sub>i</sub>|. (6)</p>

Единственное отличие от метода Арнольди заключается в том, что <i>T<sub>m</sub></i> является симметричной тридегональной матрицей, и, следовательно, существует больше возможных методов для вычисления ее собственных пар.

Алгоритм Ланцоша в конечной арифметике сталкивается с проблемами, связанными с потерей ортогональности вектора Ланцоша, что приводит к появлению спurious eigenvalues (ложных собственных значений) и множественных копий значений Ритца. Чтобы решить эти проблемы и поддерживать эффективность алгоритма, вводится концепция рестарта. 
К тому же полная ортогонализация требует хранения всех векторов Ланцоша в памяти и увеличивает вычислительные затраты с увеличением числа шагов. Рестарт ограничивает количество хранимых векторов и, таким образом, снижает затраты на память и вычисления.

Далее расмотрим вариант алгоритма с перезапуском.

## Алгоритм 3 (Ланцош с дефляцией)
(Lanczos Methods in SLEPc страница 8)
**Вход**: Матрица A</code>, количество шагов m, матрицы V<sub>k</sub> - матрица V после к-го шага алгоритма, T<sub>k</sub> - матрица T после к-го шага алгоритма, при k &lt; m, и начальный вектор v<sub>k+1</sub> с нормой 1  
**Выход**: (V<sub>m</sub>, T<sub>m</sub>, v<sub>m+1</sub>, &beta;<sub>m+1</sub>)</code> так, что AV<sub>m</sub> - V<sub>m</sub>T<sub>m</sub> = &beta;<sub>m+1</sub> v<sub>m+1</sub> e<sup>*</sup><sub>m</sub>

Для j = k + 1, &hellip;, m
1. u<sub>j+1</sub> = Av<sub>j</sub>
2. Ортогонализовать u<sub>j+1</sub> относительно V<sub>j</sub> (получая &alpha;<sub>j</sub>)
3. &beta;<sub>j+1</sub> = &#124;u<sub>j+1</sub>&#124;<sub>2</sub>
4. Если &beta;<sub>j+1</sub> = 0, остановиться
5. v<sub>j+1</sub> = u<sub>j+1</sub> / &beta;<sub>j+1</sub>
   
*Примечание*: Алгоритм 3 вычисляет только последние <code>m - k</code> столбцов матриц V<sub>m</sub> и T<sub>m</sub>, активную часть факторизации. Начальный вектор в данном случае — это v<sub>k+1</sub>. Операции в цикле схожи с Алгоритмом 2, однако ортогонализация обязательно включает заблокированные векторы Ланцоша (дефляция).

## Алгоритм 4 (Ланцош с явным перезапуском)
(Lanczos Methods in SLEPc страница 8)
**Вход**: Матрица A, начальный вектор v<sub>1</sub>, и размер подпространства m  
**Выход**: Частичное собственное разложение AV<sub>k</sub> = V<sub>k</sub>&Theta;<sub>k</sub>, где &Theta;<sub>k</sub> = diag(&theta;<sub>1</sub>, &hellip;, &theta;<sub>k</sub>)

1. Нормализовать v<sub>1</sub>
2. Инициализировать V<sub>m</sub> = [v<sub>1</sub>], k = 0

*Цикл перезапуска*
1. Выполнить m - k шагов Ланцоша с дефляцией (Алгоритм 3)
2. Вычислить собственные пары матрицы T<sub>m</sub>: T<sub>m y<sub>i</sub> = y<sub>i</sub> &theta;<sub>i</sub>
3. Оценить нормы остаточных членов, &tau;<sub>i</sub> = &beta;<sub>m+1</sub> |e<sup>*</sup><sub>m</sub> y<sub>i</sub>|
4. Зафиксировать сходимые собственные пары, обновить значение k
5. V<sub>m</sub> = V<sub>m</sub> Y

*Примечание*: В Алгоритме 4 ведущая подматрица <code>T</code>, соответствующая зафиксированным векторам, является диагональной, поэтому некоторые операции можно пропустить для этой части.
Потеря ортогональности в контексте перезапуска метода Ланцоша

В методе Ланцоша с перезапуском также необходимо учитывать потерю ортогональности. В случае использования простой схемы явного перезапуска (Алгоритм 4) можно безопасно применять любые из методов, описанных в разделе 2, так как полная ортогональность векторов Ланцоша не требуется для корректного выполнения перезапуска. Однако в случае локальной ортогонализации следует учитывать следующие моменты:

- Поскольку значение m (максимально допустимая размерность подпространства) обычно значительно меньше, чем n (размерность матрицы), эвристические подходы, предложенные Каллумом и Уиллоуби [1985], не могут быть применены. Поэтому следует использовать другой метод для отбраковки ложных собственных значений, а также для удаления избыточных дубликатов. Один из возможных подходов заключается в явном вычислении нормы остатка для каждой сходимой собственной пары, а затем, среди правильных значений, принятии только первой копии (это подробнее объясняется в разделе 3).

- Вектор для перезапуска должен быть явно ортогонализован относительно зафиксированных векторов.
## Имплементация ##
Имплементация алгоритма Ланцоша доступна в SLEPc, начиная с версии 2.3.0. Эта реализация основана на Алгоритме 4 и включает все различные методы для обработки потери ортогональности.

Алгоритм 5 (Ланцош с явным перезапуском и различными методами ортогонализации)
(Lanczos Methods in SLEPc страница 12)
**Вход**: Матрица <code>A</code>, начальный вектор <code>v<sub>1</sub></code>, и размер подпространства <code>m</code>  
**Выход**: Частичное собственное разложение <code>AV<sub>k</sub> = V<sub>k</sub>&Theta;<sub>k</sub></code>, где <code>&Theta;<sub>k</sub> = diag(&theta;<sub>1</sub>, &hellip;, &theta;<sub>k</sub>)</code>

1. Нормализовать <code>v<sub>1</sub></code>
2. Инициализировать <code>V<sub>m</sub> = [v<sub>1</sub>]</code>, <code>k = 0</code>

*Цикл перезапуска*
1. Для <code>j = k + 1, &hellip;, m</code>:
   - <code>u<sub>j+1</sub> = A v<sub>j</sub></code>
   - (*) Ортогонализовать <code>u<sub>j+1</sub></code> относительно <code>[V<sub>k</sub>, v<sub>j−1</sub>, v<sub>j</sub>]</code> (получая <code>&alpha;<sub>j</sub></code>)
   - (*) Определить множество <code>S</code> векторов Ланцоша
   - (*) Ортогонализовать <code>u<sub>j+1</sub></code> относительно <code>S</code>
   - <code>&beta;<sub>j+1</sub> = |u<sub>j+1</sub>|<sub>2</sub></code> (если <code>&beta;<sub>j+1</sub> = 0</code>, остановить)
   - <code>v<sub>j+1</sub> = u<sub>j+1</sub> / &beta;<sub>j+1</sub></code>
2. Вычислить собственные пары матрицы <code>T<sub>m</sub></code>: <code>T<sub>m</sub> y<sub>i</sub> = y<sub>i</sub> &theta;<sub>i</sub></code>
3. Оценить нормы остаточных членов: <code>&tau;<sub>i</sub> = &beta;<sub>m+1</sub> |e<sup>*</sup><sub>m</sub> y<sub>i</sub>|</code>
4. (*) Проверить на наличие ложных собственных значений
5. Зафиксировать сходимые собственные пары, обновить значение <code>k</code>
6. <code>V<sub>m</sub> = V<sub>m</sub> Y</code>

Определение множества <code>S</code> в зависимости от стратегии ортогонализации

- В случае локальной ортогонализации <code>S = &empty;</code>, поэтому вторую ортогонализацию не выполняют.
- Полная ортогонализация эквивалентна <code>S = [v<sub>k+1</sub>, v<sub>k+2</sub>, &hellip;, v<sub>j−2</sub>]</code>. В контексте параллельного исполнения обе ортогонализации, показанные в Алгоритме 5, выполняются как одна операция.
- При периодической и частичной ортогонализации рекуррентное вычисление <code>&omega;<sub>j</sub></code> производится на каждом шаге Ланцоша. Заметим, что вычислительные затраты при этом незначительны.
- При селективной ортогонализации реализация SLEPc явно вычисляет собственные значения и собственные векторы тридиагональной матрицы <code>T<sub>k</sub></code>, вместе с оценками ассоциированных норм остаточных членов, на каждом шаге Ланцоша. Отметим, что для умеренно больших <code>k</code> это может быть вычислительно затратным.

Проверка на наличие ложных собственных значений

Эта проверка в конце Алгоритма 5 необходима только для локальной ортогонализации. Стратегия заключается в следующем: собственные значения, которые повторяются в текущей тридиагональной матрице <code>T<sub>m</sub></code> после перезапуска, просто отбрасываются, исходя из предположения, что, если они действительно повторяются, то они появятся в следующем перезапуске. Для остальных собственных значений явно вычисляется истинная норма остатка <code>&#124;A x<sub>i</sub> − &lambda;<sub>i</sub>x<sub>i</sub>&#124;<sub>2</sub></code>, чтобы гарантировать, что принимаются только те пары, у которых норма остатка лежит в пределах допустимой погрешности.

Теперь перейдем к описанию кода
нас интересуют три функции: SVDTwoSideLanczos,SVDOneSideLanczos и  SVDSolve_Lanczos
начнём сначала
### SVDTwoSideLanczos ###
<p><code>PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt k,PetscInt *n,PetscBool *breakdown)</code></p>
тут alpha и beta: массивы в которую будут сохраняться элементы, связанные с сингулярными значениями. V,U: блоки векторов для ортонормализации, k: текущий идекс итерации ,n: указатель на количество итераицй ,breakdown: указатель на флаг, который указывает, произошел ли сбой из-за линейной зависимости.</p>
<p><code>PetscInt       i;
Vec            u,v;
PetscBool      lindep=PETSC_FALSE;
PetscFunctionBegin;
</code></p>
Тут объявляются переменные и макрос для начала выполнения кода в функции PETSc.
<p><code>PetscCall(BVGetColumn(svd->V,k,&v));
  PetscCall(BVGetColumn(svd->U,k,&u));
  PetscCall(MatMult(svd->A,v,u));
  PetscCall(BVRestoreColumn(svd->V,k,&v));
  PetscCall(BVRestoreColumn(svd->U,k,&u));
  PetscCall(BVOrthonormalizeColumn(svd->U,k,PETSC_FALSE,alpha+k,&lindep));
  if (PetscUnlikely(lindep)) {
    *n = k;
    if (breakdown) *breakdown = lindep;
    PetscFunctionReturn(PETSC_SUCCESS);
  }  
</code></p>
тут происходит извлечение к-го столбца из матриц V и U, извлечённые слобцы записываются в переменные v и u
Далее идёт умножение матрицы А на эти вектора, после чего матрицы V и U приходят в исходное состояние. После чего выполняется ортонормализация к-го столбца матрицы U. результат сохранятеся в массив alpha, lindep указывает произошла ли линейная зависимость, если да, то итерации прекращаются, в переменной n сохранятеся количество успешыных итераций и если был передан флаг breakdown, он устанавливается на true
<p><code>for (i=k+1;i<*n;i++) {
    PetscCall(BVGetColumn(svd->V,i,&v));
    PetscCall(BVGetColumn(svd->U,i-1,&u));
    PetscCall(MatMult(svd->AT,u,v));
    PetscCall(BVRestoreColumn(svd->V,i,&v));
    PetscCall(BVRestoreColumn(svd->U,i-1,&u));
    PetscCall(BVOrthonormalizeColumn(svd->V,i,PETSC_FALSE,beta+i-1,&lindep));
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }
    PetscCall(BVGetColumn(svd->V,i,&v));
    PetscCall(BVGetColumn(svd->U,i,&u));
    PetscCall(MatMult(svd->A,v,u));
    PetscCall(BVRestoreColumn(svd->V,i,&v));
    PetscCall(BVRestoreColumn(svd->U,i,&u));
    PetscCall(BVOrthonormalizeColumn(svd->U,i,PETSC_FALSE,alpha+i,&lindep));
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }
  }
</code></p>
тут выполняются итерации алгоритма Ланцоша с i = k + 1 до n-1.
Сначала извлекаются столбцы из матриц V и U с индексами i и i-1,
Выполняется умножение транспонированной матрицы A на вектор u, результат сохраняется в v,
матрицы V U возвращаются к первоначальному виду,
Выполняется ортонормализация выполняется для V, результат сохраняется массив beta,
Если возникает линейная зависимость цикл завершается.
После выполняются аналогичные действия для матрицы A.
<p><code>if (!lindep) {
    PetscCall(BVGetColumn(svd->V,*n,&v));
    PetscCall(BVGetColumn(svd->U,*n-1,&u));
    PetscCall(MatMult(svd->AT,u,v));
    PetscCall(BVRestoreColumn(svd->V,*n,&v));
    PetscCall(BVRestoreColumn(svd->U,*n-1,&u));
    PetscCall(BVOrthogonalizeColumn(svd->V,*n,NULL,beta+*n-1,&lindep));
  }
  if (breakdown) *breakdown = lindep;
  PetscFunctionReturn(PETSC_SUCCESS);
}
</code></p>
После окончания всех итераций если линйеной зависимости нет, выполняетсся дополнительная ортогонализация для последнего столбца. В самом конце функция возвпращает статус выполнения и устанавливается значение флага breakdown на true.

### SVDOneSideLanczos ###
Входные данные почти теже, добавляется u_1 это вектор который будет спользоваться для хранения промежуточных значений
<p><code>static PetscErrorCode SVDOneSideLanczos(SVD svd, PetscReal *alpha, PetscReal *beta, BV V, Vec u, Vec u_1, PetscInt k, PetscInt n, PetscScalar *work)
{
  PetscInt i, bvl, bvk;
  PetscReal a, b;
  Vec z, temp;

  PetscFunctionBegin;
  PetscCall(BVGetActiveColumns(V, &bvl, &bvk));
</code></p>
Объявляются переменные и задаётся срез столбцов с которым будем работать.
<p><code>PetscCall(BVGetColumn(V, k, &z));
  PetscCall(MatMult(svd->A, z, u));
  PetscCall(BVRestoreColumn(V, k, &z));
</code></p>
Умножение к-го стоблца матрицы V на матрицу А
<p><code>for (i = k + 1; i < n; i++) {
    PetscCall(BVGetColumn(V, i, &z));
    PetscCall(MatMult(svd->AT, u, z));
    PetscCall(BVRestoreColumn(V, i, &z));
    PetscCall(VecNormBegin(u, NORM_2, &a));
    PetscCall(BVSetActiveColumns(V, 0, i));
    PetscCall(BVDotColumnBegin(V, i, work));
    PetscCall(VecNormEnd(u, NORM_2, &a));
    PetscCall(BVDotColumnEnd(V, i, work));
    PetscCall(VecScale(u, 1.0 / a));
    PetscCall(BVMultColumn(V, -1.0 / a, 1.0 / a, i, work));
</code></p>
Цикл, который идет от k + 1 до n, и на каждом шаге выполняются следующие действия:
Извлекается i-я колонка из V, умножается на транспонированную матрицу А.
Далее происходит нормализация вектора U с помощью VecNormBegin и VecNormEnd для вычисления его нормы. Затем Вектор нормализуется: каждый элемент вектора U делится на его норму.
<p><code>PetscCall(BVDotColumn(V, i, work));
    PetscCall(BVMultColumn(V, -1.0, 1.0, i, work));
    PetscCall(BVNormColumn(V, i, NORM_2, &b));
    PetscCheck(PetscAbsReal(b) > 10 * PETSC_MACHINE_EPSILON,     
    PetscObjectComm((PetscObject)svd), PETSC_ERR_PLIB, "Recurrence generated a zero vector; use a two-sided variant");
    PetscCall(BVScaleColumn(V, i, 1.0 / b));
</code></p>
Ортогонализация i-того столбца.
Если после ортогонализации вектор становится очень малым (меньше определенного порога), возникает ошибка, и предлагается использовать двухсторонний метод.
<p><code>PetscCall(BVGetColumn(V, i, &z));
    PetscCall(MatMult(svd->A, z, u_1));
    PetscCall(BVRestoreColumn(V, i, &z));
    PetscCall(VecAXPY(u_1, -b, u));
    alpha[i - 1] = a;
    beta[i - 1] = b;
    temp = u;
    u = u_1;
    u_1 = temp;
  }
</code></p>
Обновляется вектор u_1 с помощью матричного умножения.
векторы u и u_1 меняются местами.
Коэффициенты a и b сохраняются в массивы.
<p><code>PetscCall(BVGetColumn(V, n, &z));
  PetscCall(MatMult(svd->AT, u, z));
  PetscCall(BVRestoreColumn(V, n, &z));
  PetscCall(VecNormBegin(u, NORM_2, &a));
  PetscCall(BVDotColumnBegin(V, n, work));
  PetscCall(VecNormEnd(u, NORM_2, &a));
  PetscCall(BVDotColumnEnd(V, n, work));
  PetscCall(VecScale(u, 1.0 / a));
  PetscCall(BVMultColumn(V, -1.0 / a, 1.0 / a, n, work));
  PetscCall(BVDotColumn(V, n, work));
  PetscCall(BVMultColumn(V, -1.0, 1.0, n, work));
  PetscCall(BVNormColumn(V, i, NORM_2, &b));

  alpha[n - 1] = a;
  beta[n - 1] = b;
  PetscCall(BVSetActiveColumns(V, bvl, bvk));
  PetscFunctionReturn(PETSC_SUCCESS);
}
</code></p>
Для последней итерации нормализуется вектор и выполняется ортогонализация с использованием транспонированной матрицы A. В конце восстанавливается активный диапазон столбцов в матрице V.

### SVDSolve_Lanczos ###
Эта функция реализует основной алгоритм для решения задачи сингулярных значений методом Ланцоша.
<p><code>PetscCall(DSGetLeadingDimension(svd->ds, &ld));
  PetscCall(PetscMalloc2(ld, &w, svd->ncv, &swork));
  if (lanczos->oneside) {
    PetscCall(MatCreateVecs(svd->A, NULL, &u));
    PetscCall(MatCreateVecs(svd->A, NULL, &u_1));
}
  if (!svd->nini) {
    PetscCall(BVSetRandomColumn(svd->V, 0));
    PetscCall(BVOrthonormalizeColumn(svd->V, 0, PETSC_TRUE, NULL, NULL));
}
</code></p>
Получается ведущая размерность и выделяется память для массивов w и swork.
Создаются вектора для одностороннего метода.
Если начальный вектор не задан, инициализируется случайным вектором и ортонормализуется.
Далее идёт основной итерационный цикл:
<p><code>while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;
    nv = PetscMin(svd->nconv + svd->mpd, svd->ncv);
    PetscCall(DSGetArrayReal(svd->ds, DS_MAT_T, &alpha));
    beta = alpha + ld;
    if (lanczos->oneside) {
    PetscCall(SVDOneSideLanczos(svd, alpha, beta, svd->V, u, u_1, svd->nconv, nv, swork));
    } else {
    PetscCall(SVDTwoSideLanczos(svd, alpha, beta, svd->V, svd->U, svd->nconv, &nv, NULL));
    PetscCall(BVSetActiveColumns(svd->U, svd->nconv, nv));
    }    
</code></p>
Устанавливется размерность, после чего выделяется память под массивы alpha и beta.
Вызывается односторонний или двусторонний алгоритм Ланцоша в зависимости от выбранного метода.
Далее идёт сингулярное разложения матрицы T:
<p><code>PetscCall(DSSetDimensions(svd->ds, nv, svd->nconv, 0));
  PetscCall(DSSVDSetDimensions(svd->ds, nv));
  PetscCall(SVDKrylovConvergence(svd, PETSC_FALSE, svd->nconv, nv - svd->nconv, 1.0, &k));
  if (svd->reason == SVD_CONVERGED_ITERATING) {
    if (k < nv) {
        PetscCall(DSGetArray(svd->ds, DS_MAT_V, &P));
        for (j = svd->nconv; j < nv; j++) swork[j - svd->nconv] = PetscConj(P[j + k * ld]);
        PetscCall(DSRestoreArray(svd->ds, DS_MAT_V, &P));
        PetscCall(BVMultColumn(svd->V, 1.0, 0.0, nv, swork));
    } else {
        PetscCall(BVSetRandomColumn(svd->V, nv));
        PetscCall(BVOrthonormalizeColumn(svd->V, nv, PETSC_FALSE, NULL, NULL));
    }
  }
  PetscCall(DSGetMat(svd->ds, DS_MAT_V, &V));
  PetscCall(BVMultInPlace(svd->V, V, svd->nconv, k));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&u_1));
  PetscCall(PetscFree2(w, swork));
</code></p>
Матрица Т проходит сингулярное разложение, проверяется сходимость, если сходимость не доснигнута, генерируется новый начальный вектор,затем вычесляются вингулярные вектора и очищается память.
## Список литературы ##
<p>Lanczos and the Riemannian SVD in information retrieval applications-Fierro-Jiang-2005 страница 359</p>
<p><a href="https://slepc.upv.es/documentation/reports/str5.pdf">Lanczos Methods in SLEPc</a></p>
<p><a href="https://slepc.upv.es/documentation/reports/str8.pdf">Restarted Lanczos Bidiagonalization for the SVD in SLEPc</a></p>
