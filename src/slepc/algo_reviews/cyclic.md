???

## Введение

* shell == оболочная == неявная матрица ???
* *Глобальные размеры матрицы* 𝑀×𝑁 представляют общее число строк (M) и столбцов (N) в матрице. Эти размеры относятся ко всей матрице, независимо от того, как она распределена между процессами.
* *Локальные размеры матрицы* (𝑚×𝑛) представляют количество строк (m) и столбцов (n), которые хранятся и обрабатываются текущим процессом MPI. Эти размеры зависят от числа процессов и способа распределения матрицы.
* Циклическая матрица - это такая блочная матрица C, которая состоит из A и $A^T$, и имеет вид:

$$
C = \begin{bmatrix}
0 & A \\
A^T & 0
\end{bmatrix}
$$

* Неявная матрица - это абстрактное представление матрицы, в котором её значения или структура известны, но не представлены в памяти как конкретные данные.

| **Примеры неявных матриц:**
1. Разреженные матрицы - такие, у которых большинство элементов равны нулю. Вместо того чтобы хранить всю матрицу целиком, используют специальные структуры данных, такие как списки смежностей или формат COO (координатный формат), которые хранят только ненулевые элементы вместе с их индексами. Таким образом, матрица становится неявной, поскольку большая часть её данных не хранится явно.

2. Квантование матриц - Некоторые методы квантования представляют матрицы в сжатом виде, используя ограниченное количество бит для каждого элемента. Это позволяет уменьшить размер матрицы за счёт точности, и в результате мы получаем неявное представление матрицы.

3. Алгоритмическое задание матриц.
Иногда матрицы могут задаваться алгоритмами, которые генерируют элементы по мере необходимости. Например, если матрица является результатом некоторого процесса, её элементы могут вычисляться динамически, а не храниться в памяти заранее.

4. Использование функций для генерации элементов.
В некоторых случаях элементы матрицы можно задать функцией, которая возвращает значение элемента по его индексу. Такой подход особенно полезен, когда необходимо работать с очень большими матрицами или с бесконечными последовательностями.

Преимущества неявного представления матриц:
 - Экономия памяти: Для больших матриц хранение всех элементов может потребовать значительных ресурсов, тогда как неявная форма позволяет избежать этого.
 - Более эффективное выполнение операций: Некоторые операции над матрицами могут выполняться быстрее, если они реализованы через функции, а не через прямое обращение к каждому элементу.

Недостатки:
 - Ограничения на некоторые виды операций: Не все операции можно эффективно выполнять с неявно заданными матрицами. Например, сложение двух неявно определённых матриц может требовать их полного раскрытия.
 - Сложность реализации: Работа с неявным представлением требует дополнительных усилий при разработке и тестировании алгоритмов.
---
### непроверено: про SVD_CYCLIC и SVD_CYCLIC_SHELL ???

В коде библиотеки SLEPc структуры SVD_CYCLIC и SVD_CYCLIC_SHELL используются для реализации и хранения данных, необходимых для решения сингулярных задач (SVD, Singular Value Decomposition) методом циклической структуры. Они представляют собой два уровня абстракции, предназначенных для организации вычислений, хранения промежуточных данных и эффективной работы с матрицами в различных конфигурациях (включая GPU).
1. SVD_CYCLIC: общий уровень данных

Эта структура используется для хранения общих параметров и объектов, связанных с циклическим методом SVD.

Поле explicitmatrix (bool) в структуре SVD_CYCLIC задаёт, каким образом будет создана матрица CC:

Если *explicitmatrix = PETSC_TRUE*:
Матрица создаётся явно, как полный объект в памяти. Все элементы CC вычисляются и хранятся в виде готовой матрицы.  
Если *explicitmatrix = PETSC_FALSE*:
Матрица CC не хранится явно. Вместо этого используется shell-матрица, которая предоставляет доступ к операциям над CC (например, матрично-векторное умножение) без полного хранения элементов. Это позволяет сэкономить память.

2. SVD_CYCLIC_SHELL: детали реализации

Эта структура используется для реализации shell-матрицы (расширенной кросс-матрицы) при работе с методом. Она позволяет реализовывать операции над матрицей "на лету", экономя память.
Поля:

Зачем нужны эти структуры?
Для чего используется **SVD_CYCLIC**?

Хранение параметров и объектов, управляющих работой метода.
Управление, используется ли явное или неявное представление матрицы CC.
Интеграция с решателем собственных значений EPS для обработки промежуточных задач.

Для чего используется **SVD_CYCLIC_SHELL**?

Реализация матрицы CC как shell-объекта (вычисления "на лету"), чтобы уменьшить использование памяти.
Хранение всех вспомогательных данных для операций с матрицей CC, включая:
Векторы для матрично-векторных операций.
Флаги и параметры для обработки GPU.
Рабочие векторы для управления несоответствиями памяти на GPU.



### Generalized singular value decomposition ???
В линейной алгебре обобщенная сингулярная декомпозиция (GSVD) - это название двух различных методов, основанных на сингулярной декомпозиции (SVD). Эти две версии отличаются тем, что одна версия разлагает две матрицы (что-то вроде SVD более высокого порядка или тензорной матрицы), а другая версия использует набор ограничений, наложенных на левый и правый сингулярные векторы SVD с одной матрицей.

**Про обобщенную задачу:** В стандартной задаче сингулярного разложения ищутся сингулярные значения и векторы матрицы A, такие что: 
$A = U * \Sigma * V$
#### В обобщённой задаче сингулярного разложения решается проблема вида:
$A x = \sigma B y, \quad A^T y = \sigma B^T x$   
где:
- $A$ и $B$ — матрицы, задающие обобщённую структуру задачи;
- $\sigma$ — сингулярные значения;
- $x$ и $y$ — обобщённые сингулярные векторы.
---
* cross product (Кросс-произведение) == векторное произведение  

В этом файле (пример SVDCyclicGetECrossMat) матрица кросс произведения это $A^T A$, где элементы вычисляются простым перемножением этих двух матриц. Представить это в краткой форме можно как:

$$ [A^T A]_{ij} = \sum_{k=1}^M A_{ki} \cdot A_{kj} $$

* extended (Расширенная) матрица С задаётся как: 

$$
C = \begin{bmatrix}
I_m & 0 \\
0 & A^T A
\end{bmatrix}
$$

где $I_m$ - единичная матрица размера m на m

## Описание алгоритма 

### ```SVDCyclicGetCyclicMat(SVD svd,Mat A,Mat AT,Mat *C)``` 
функция предназначена для вычисления циклической матрицы 𝐶 на основе переданных параметров A и $A^T$. Поддерижвается как CUDA и HIP (параллельное распределение процессов), так CUDA и HIP одновременно.

1. Приводит указатель svd->data к типу SVD_CYCLIC и объявление переменных:
* ctx — контекст для работы с неявной-матрицей.
* i — индекс.
* M, N — глобальные размеры матрицы 𝐴.
* m, n — локальные размеры матрицы 𝐴 для текущего процесса.
* Istart, Iend — начало и конец диапазона индексов для текущего процесса.
* vtype — тип векторов, используемых в матрице.
* Zm, Zn — вспомогательные матрицы для создания циклической матрицы.

2. Провереятся используется ли CUDA или HIP, так же определяется вспомогательные параметры (диапозоны и количество процессов).
3. Определяются глобальные и локальные размеры матрицы A.
4. Если матрица задана явным видом, то:
- Проверяется для объекта SVD: существует ли опция явного транспонирования матрицы, иначе выдается ошибка.
- Создание вспомогательной матрицы $Z_m$, где задаются её локальные и глобальные размеры.
- Определяется определяет диапазон строк, принадлежащих текущему процессу. Затем Цикл проходит по строкам в этом диапазоне и устанавливает диагональные элементы в 0.
​- Происходит сборка матрица и завершается.
- Аналогичная иницилиализация матрицы $Z_n$.
- Затем строится матрица циклическая матрица C.
- Удаление временных матриц $Z_n$ и $Z_m$
5. Иначе, если матрица заданы неявным видом, то:
- Создается объект ctx, который хранится хранящая указатели матрицу на A, $A^T$ и флаг swapped.
- Создание shell-неявной матрица согласно размерности и контексту.
- Создаются пустые векторы, соответствующие размерности матрицы A, которые могут использоваться, например, для временного хранения промежуточных результатов операций с матрицей A.
- Создание неявной матрицы на основе размерностей и матриц A и $A^T$, учитывая распараллеливание.
- Установление базовых операций для матрицы C.
6. Определяется поддержка CUDA(GPU), HIP и одновременная поддержка.
 
```
static PetscErrorCode SVDCyclicGetCyclicMat(SVD svd,Mat A,Mat AT,Mat *C)
{
  SVD_CYCLIC       *cyclic = (SVD_CYCLIC*)svd->data;
  SVD_CYCLIC_SHELL *ctx;
  PetscInt         i,M,N,m,n,Istart,Iend;
  VecType          vtype;
  Mat              Zm,Zn;
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  PetscBool        gpu;
  const PetscInt   *ranges;
  PetscMPIInt      size;
#endif

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));

  if (cyclic->explicitmatrix) {
    PetscCheck(svd->expltrans,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
    PetscCall(MatCreate(PetscObjectComm((PetscObject)svd),&Zm));
    PetscCall(MatSetSizes(Zm,m,m,M,M));
    PetscCall(MatSetFromOptions(Zm));
    PetscCall(MatGetOwnershipRange(Zm,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(Zm,i,i,0.0,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Zm,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Zm,MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)svd),&Zn));
    PetscCall(MatSetSizes(Zn,n,n,N,N));
    PetscCall(MatSetFromOptions(Zn));
    PetscCall(MatGetOwnershipRange(Zn,&Istart,&Iend));
    for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(Zn,i,i,0.0,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Zn,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreateTile(1.0,Zm,1.0,A,1.0,AT,1.0,Zn,C));
    PetscCall(MatDestroy(&Zm));
    PetscCall(MatDestroy(&Zn));
  } else {
    PetscCall(PetscNew(&ctx));
    ctx->A       = A;
    ctx->AT      = AT;
    ctx->swapped = svd->swapped;
    PetscCall(MatCreateVecsEmpty(A,&ctx->x2,&ctx->x1));
    PetscCall(MatCreateVecsEmpty(A,&ctx->y2,&ctx->y1));
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,ctx,C));
    PetscCall(MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Cyclic));
    PetscCall(MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_Cyclic));
#if defined(PETSC_HAVE_CUDA)
    PetscCall(PetscObjectTypeCompareAny((PetscObject)(svd->swapped?AT:A),&gpu,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,""));
    if (gpu) PetscCall(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic_CUDA));
    else
#elif defined(PETSC_HAVE_HIP)
    PetscCall(PetscObjectTypeCompareAny((PetscObject)(svd->swapped?AT:A),&gpu,MATSEQAIJHIPSPARSE,MATMPIAIJHIPSPARSE,""));
    if (gpu) PetscCall(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic_HIP));
    else
#endif
      PetscCall(MatShellSetOperation(*C,MATOP_MULT,(void(*)(void))MatMult_Cyclic));
    PetscCall(MatGetVecType(A,&vtype));
    PetscCall(MatSetVecType(*C,vtype));
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
    if (gpu) {
      /* check alignment of bottom block */
      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ctx->x1),&size));
      PetscCall(VecGetOwnershipRanges(ctx->x1,&ranges));
      for (i=0;i<size;i++) {
        ctx->misaligned = (((ranges[i+1]-ranges[i])*sizeof(PetscScalar))%16)? PETSC_TRUE: PETSC_FALSE;
        if (ctx->misaligned) break;
      }
      if (ctx->misaligned) {  /* create work vectors for MatMult */
        PetscCall(VecDuplicate(ctx->x2,&ctx->wx2));
        PetscCall(VecDuplicate(ctx->y2,&ctx->wy2));
      }
    }
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### MatGetDiagonal_ECross(Mat B,Vec d) - фукнция, которая предназначена для вычисления главной диагонали матрицы B, заданная неявным образом.
1. Объявление переменных:
* ctx: контекст shell-матрицы, содержащий данные для её обработки.
* pd: указатель на массив данных вектора d.
* len: длина массивов (в формате MPI).
* mn, m, n, N: размеры локальных и глобальных матриц и векторов.
* start, end: диапазон строк, принадлежащих текущему процессу.
* ncols: количество столбцов.
* work1, work2: временные массивы для вычислений.
* diag: указатель на массив данных диагонального вектора.
2. Получение контекста неявной матрицы A ($A^T$, вспомогательные вектора и т д) в параметр ctx. Получение локального количества n столбцов матрицы А, локального размера mn вектора d. Вычисление локального числа строк: m=mn-n.
3. Получение указателя на вектор pd и привязка его к веткору ctx->y1. Иницилиализация ветора *pd = ctx->y1 (все значения вектора равны 1). Затем отмена привязки.
4. Привязка второй части массива pd к вектору ctx->y2.
5. Если не является диагональной:
6. Создание нового дубликата для ctx->y2.
7. Инициализация временных массивов work1 и work2
8. Проверяется не является ли матрица транспонированной, то мы получаем диапозон для текущего процесса HIP(MPI) для транспонированной матрица $A^T$.
- Цикл, котороый проходит по всем строкам матрицы $A^T$, входящих в диапозон текущего процесса, MatGetRow извлекает данные текущей строки: количество столбцов (ncols) и значения элементов строки (vals).
- Квадраты значений строки vals[j] суммируются в work1[i]. После обработки строки данные возвращаются через MatRestoreRow.
9. Если явтяется является транспонированной матрицей, то мы делаем все то же самое из пункта 8, только для матрицы A, так как она уже является транспонированной.
10. Затем идет работы объединение данных процессов HIP(MPI): преобразование локальных размеров в глобальные размеры, объединения результатов вычисленных значений каждого процесса в массив wokr2.
11. Идет получения данных для текущего процесса HIP(MPI), происходит цикл в диапозоне, полученным для текущего процесса, который делает записывает локальную часть вектора work2 в локальную часть вектора diag. 
12. Затем идет удаление временного объекта work2.
13. Копирует данные из ctx->diag в вектор ctx->y2. VecResetArray отменяет связь массива данных с вектором ctx->y2. VecRestoreArrayWrite завершает запись
```
static PetscErrorCode MatGetDiagonal_ECross(Mat B,Vec d)
{
  SVD_CYCLIC_SHELL  *ctx;
  PetscScalar       *pd;
  PetscMPIInt       len;
  PetscInt          mn,m,n,N,i,j,start,end,ncols;
  PetscScalar       *work1,*work2,*diag;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(MatGetLocalSize(ctx->A,NULL,&n));
  PetscCall(VecGetLocalSize(d,&mn));
  m = mn-n;
  PetscCall(VecGetArrayWrite(d,&pd));
  PetscCall(VecPlaceArray(ctx->y1,pd));
  PetscCall(VecSet(ctx->y1,1.0));
  PetscCall(VecResetArray(ctx->y1));
  PetscCall(VecPlaceArray(ctx->y2,pd+m));
  if (!ctx->diag) {
    /* compute diagonal from rows and store in ctx->diag */
    PetscCall(VecDuplicate(ctx->y2,&ctx->diag));
    PetscCall(MatGetSize(ctx->A,NULL,&N));
    PetscCall(PetscCalloc2(N,&work1,N,&work2));
    if (ctx->swapped) {
      PetscCall(MatGetOwnershipRange(ctx->AT,&start,&end));
      for (i=start;i<end;i++) {
        PetscCall(MatGetRow(ctx->AT,i,&ncols,NULL,&vals));
        for (j=0;j<ncols;j++) work1[i] += vals[j]*vals[j];
        PetscCall(MatRestoreRow(ctx->AT,i,&ncols,NULL,&vals));
      }
    } else {
      PetscCall(MatGetOwnershipRange(ctx->A,&start,&end));
      for (i=start;i<end;i++) {
        PetscCall(MatGetRow(ctx->A,i,&ncols,&cols,&vals));
        for (j=0;j<ncols;j++) work1[cols[j]] += vals[j]*vals[j];
        PetscCall(MatRestoreRow(ctx->A,i,&ncols,&cols,&vals));
      }
    }
    PetscCall(PetscMPIIntCast(N,&len));
    PetscCallMPI(MPIU_Allreduce(work1,work2,len,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)B)));
    PetscCall(VecGetOwnershipRange(ctx->diag,&start,&end));
    PetscCall(VecGetArrayWrite(ctx->diag,&diag));
    for (i=start;i<end;i++) diag[i-start] = work2[i];
    PetscCall(VecRestoreArrayWrite(ctx->diag,&diag));
    PetscCall(PetscFree2(work1,work2));
  }
  PetscCall(VecCopy(ctx->diag,ctx->y2));
  PetscCall(VecResetArray(ctx->y2));
  PetscCall(VecRestoreArrayWrite(d,&pd));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### SVDSetUp_Cyclic(SVD svd) - функция, которая выполняет настройку параметров для сингулярного разложения в циклическом методе:
#### - Определяет и создаёт матрицы, необходимые для алгоритма.
#### - Настраивает параметры EPS.
#### - Настраивает параметры точности, размеры подпространств и критерии сходимости.
#### - Обеспечивает настройку для обобщенных (GSVD = Generalized singular value decomposition) и гиперболических задач.

1. Приводит указатель svd->data к типу SVD_CYCLIC и объявление переменных:
* ctx — контекст для работы с неявной-матрицей.
* M, N — глобальные размеры матрицы 𝐴.
* new - количество собственных значений для вычисления;
* ncv - номер базисного вектора;
* mpd - максимальный размер проектируемой задачи;
* m, n — локальные размеры матрицы 𝐴 для текущего процесса.
* nev, ncv, mpd - параметры вычисления собственных значений (число собственных значений, размер подпространства, максимальная размерность подпространства).
* tol - точность вычислений.
* maxit - максимальное число итераций.
* *isa,*oa, *va - указатели на массивы данных (например, элементы векторов).
* ptype - тип задачи для EPS;
* trackall - должны ли быть вычислены все остатки;
* issinv;
* v,t;
* st - объект спектрального преобразования;
* Omega - вспомогательная матрица для гиперболических задач;
* Atype;

2. Извлекаются данные из объекта svd. Считываются размеры глобальной (M, N) и локальной (m, n) матрицы svd->A.
3. Проверяется, существует ли объект EPS для решение задачи собственных значений, если нет то создается объект.
4. Если ранее создавались матрицы cyclic->C и cyclic->D, они удаляются.
5. Если является задача обобщённым типом сингулярного разложения.
- 5.1 Если требуется найти наименьшие значения то создаются вспомогательные матрицы C (на основе B и $B^T) и D(на основе A и $A^T).
- 5.2 Иначе, то же создаются матрицы C и D, только местами наоборот.
- 5.3 Оператор C используется как основной, D — как дополнительный. Проверяется, установлен ли тип задачи ptype. Если нет, то тип задается эрмитовой тип задачи.
6. Если не является обобщенным типом. И является гиперболическим типом:
- 6.1 Строит циклическую матрицу C с использованием A и ее транспонированной $A^T$. и Создает вектор v, совместимый с C, и инициализирует его единицами.
- 6.2 Предоставляется доступ к внутренним данным вектора svd->omega для чтения. Позволяет получить прямой доступ к элементам вектора ω в виде массива oa
- 6.3 Проверяется является ли матрица транспонированной, то копирует n элементов из oa в va + m, иначе копирует m элементов из oa в va.
- 6.4 Указывает, что операции с массивами завершены, и восстанавливает состояние.
- 6.5 Получает тип матрицы оператора svd->OP. Создает новую матрицу Omega с соответствующими размерами и типом. Устанавливает диагональные элементы Omega, используя вектор v.
- 6.6 Устанавливает C и Omega как операторы для решателя EPS. Устанавливает тип задачи как EPS_GHIEP (обобщенная гиперболическая задача собственных значений). Затем удаляются временные матрица Omega и вектор v.
- 6.7 Если не гиперболического типа (стандартного типа), то строит C, используя A и $A^T$. Устанавливает C как оператор для решателя EPS. Устанавливает тип задачи как EPS_HEP (эрмитова задача собственных значений).
7. Если пользователь не предоставил тип решения EPS для решателя EPS, то:
- 7.1 Если вычисляются наибольшие сингулярные значения, то Проверяет, является ли спектральное преобразование типа STSINVERT (метод сдвига-инвертирования (shift-invert)).
- 7.2 Если да, устанавливает решатель EPS для поиска собственных значений вблизи величины EPS_TARGET_MAGNITUDE.
- 7.3 Если задача гиперболическая, то ищутся собственные значения с наибольшей величиной.
- 7.4 Иначе нацеливается на наибольшие действительные собственные значения.
- 7.5 Если не наибольшее собственные значения, то Для обобщенных задач устанавливает решатель EPS для поиска наибольших действительных собственных значений.
- 7.6 Для гиперболических или стандартных задач устанавливается решатель EPS для поиска наименьших положительных действительных собственных значений.
8. Получает текущие размеры (nev, ncv, mpd) из решателя EPS. Убеждается, что число собственных значений nev не меньше двойного числа сингулярных значений svd->nsv. Обновляет nev, ncv и mpd, если они ранее не были установлены.
9. Получает текущую точность и максимальное количество итераций. Если они не установлена, используется из svd или значения по умолчанию. Устанавливает в решателе EPS.
10. Устанавливает тест сходимости для решателя EPS на основе критерия сходимости SVD (svd->conv). 
11. Для абсолютного, относительного и основанного на норме типов сходимости устанавливаются соответствующие функции.
12. Для обобщенного SVD с сходимостью по норме используется пользовательская функция сходимости EPSConv_Cyclic.
13. Вызывает ошибку для неподдерживаемых типов сходимости.
14. Передает опцию trackall из контекста SVD в решатель EPS, указывая, следует ли отслеживать все собственные значения во время итераций.
15.  Определяет, задано ли начальное подпространство для левых (svd->ISL) или правых (svd->IS) сингулярных векторов, то
- 15.1 Перебирает начальные векторы, чтобы подготовить их для решателя EPS.
- 15.2 Создает вектор v, совместимый с матрицей C. Получает доступ для записи к данным вектора v.
- 15.3 Размер k верхнего блочного ряда в построенном векторе v, который зависит от того, является ли задача обобщенной и вычисляются ли наименьшие сингулярные значения.
- 15.4 Если задан левый начальный вектор, его соответствующая часть копируется в va. Проверяются размеры. Если начальный вектор отсутствует, соответствующая часть заполняется нулями.
- 15.5 Если задан правый начальный вектор, он копируется в соответствующую позицию в va. Обеспечивается совместимость размеров векторов и матрицы. Если начальный вектор отсутствует, соответствующая часть заполняется нулями.
- 15.6 Восстанавливает доступ к v. Заменяет старый начальный вектор новым объединенным вектором v.
- 15.7 Обновляет число начальных векторов. Устанавливает начальное пространство в решателе EPS. Удаляются начальные векторы.
- 15.8 Завершает настройку решателя EPS после всех типов.
- 15.9 Получает размер подпространства ncv и максимальную размерность проекции mpd из решателя EPS. svd->ncv = минимум из M и N. 
- 15.10 Получает максимальное число итераций из решателя EPS. Устанавливается точность svd->tol, если она ранее не была установлена.
- 15.11 Указывает, что будут вычислены левые сингулярные векторы. Выделяет память для векторов решения.
```
static PetscErrorCode SVDSetUp_Cyclic(SVD svd)
{
  SVD_CYCLIC        *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt          M,N,m,n,p,k,i,isl,offset,nev,ncv,mpd,maxit;
  PetscReal         tol;
  const PetscScalar *isa,*oa;
  PetscScalar       *va;
  EPSProblemType    ptype;
  PetscBool         trackall,issinv;
  Vec               v,t;
  ST                st;
  Mat               Omega;
  MatType           Atype;

  PetscFunctionBegin;
  PetscCall(MatGetSize(svd->A,&M,&N));
  PetscCall(MatGetLocalSize(svd->A,&m,&n));
  if (!cyclic->eps) PetscCall(SVDCyclicGetEPS(svd,&cyclic->eps));
  PetscCall(MatDestroy(&cyclic->C));
  PetscCall(MatDestroy(&cyclic->D));
  if (svd->isgeneralized) {
    if (svd->which==SVD_SMALLEST) {  /* alternative pencil */
      PetscCall(MatCreateVecs(svd->B,NULL,&t));
      PetscCall(SVDCyclicGetCyclicMat(svd,svd->B,svd->BT,&cyclic->C));
      PetscCall(SVDCyclicGetECrossMat(svd,svd->A,svd->AT,&cyclic->D,t));
    } else {
      PetscCall(MatCreateVecs(svd->A,NULL,&t));
      PetscCall(SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C));
      PetscCall(SVDCyclicGetECrossMat(svd,svd->B,svd->BT,&cyclic->D,t));
    }
    PetscCall(VecDestroy(&t));
    PetscCall(EPSSetOperators(cyclic->eps,cyclic->C,cyclic->D));
    PetscCall(EPSGetProblemType(cyclic->eps,&ptype));
    if (!ptype) PetscCall(EPSSetProblemType(cyclic->eps,EPS_GHEP));
  } else if (svd->ishyperbolic) {
    PetscCall(SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C));
    PetscCall(MatCreateVecs(cyclic->C,&v,NULL));
    PetscCall(VecSet(v,1.0));
    PetscCall(VecGetArrayRead(svd->omega,&oa));
    PetscCall(VecGetArray(v,&va));
    if (svd->swapped) PetscCall(PetscArraycpy(va+m,oa,n));
    else PetscCall(PetscArraycpy(va,oa,m));
    PetscCall(VecRestoreArrayRead(svd->omega,&oa));
    PetscCall(VecRestoreArray(v,&va));
    PetscCall(MatGetType(svd->OP,&Atype));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)svd),&Omega));
    PetscCall(MatSetSizes(Omega,m+n,m+n,M+N,M+N));
    PetscCall(MatSetType(Omega,Atype));
    PetscCall(MatDiagonalSet(Omega,v,INSERT_VALUES));
    PetscCall(EPSSetOperators(cyclic->eps,cyclic->C,Omega));
    PetscCall(EPSSetProblemType(cyclic->eps,EPS_GHIEP));
    PetscCall(MatDestroy(&Omega));
    PetscCall(VecDestroy(&v));
  } else {
    PetscCall(SVDCyclicGetCyclicMat(svd,svd->A,svd->AT,&cyclic->C));
    PetscCall(EPSSetOperators(cyclic->eps,cyclic->C,NULL));
    PetscCall(EPSSetProblemType(cyclic->eps,EPS_HEP));
  }
  if (!cyclic->usereps) {
    if (svd->which == SVD_LARGEST) {
      PetscCall(EPSGetST(cyclic->eps,&st));
      PetscCall(PetscObjectTypeCompare((PetscObject)st,STSINVERT,&issinv));
      if (issinv) PetscCall(EPSSetWhichEigenpairs(cyclic->eps,EPS_TARGET_MAGNITUDE));
      else if (svd->ishyperbolic) PetscCall(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_MAGNITUDE));
      else PetscCall(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL));
    } else {
      if (svd->isgeneralized) {  /* computes sigma^{-1} via alternative pencil */
        PetscCall(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL));
      } else {
        if (svd->ishyperbolic) PetscCall(EPSSetWhichEigenpairs(cyclic->eps,EPS_TARGET_MAGNITUDE));
        else PetscCall(EPSSetEigenvalueComparison(cyclic->eps,SlepcCompareSmallestPosReal,NULL));
        PetscCall(EPSSetTarget(cyclic->eps,0.0));
      }
    }
    PetscCall(EPSGetDimensions(cyclic->eps,&nev,&ncv,&mpd));
    PetscCheck(nev==1 || nev>=2*svd->nsv,PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"The number of requested eigenvalues %" PetscInt_FMT " must be at least 2*%" PetscInt_FMT,nev,svd->nsv);
    nev = PetscMax(nev,2*svd->nsv);
    if (ncv==PETSC_DETERMINE && svd->ncv!=PETSC_DETERMINE) ncv = PetscMax(3*svd->nsv,svd->ncv);
    if (mpd==PETSC_DETERMINE && svd->mpd!=PETSC_DETERMINE) mpd = svd->mpd;
    PetscCall(EPSSetDimensions(cyclic->eps,nev,ncv,mpd));
    PetscCall(EPSGetTolerances(cyclic->eps,&tol,&maxit));
    if (tol==(PetscReal)PETSC_DETERMINE) tol = svd->tol==(PetscReal)PETSC_DETERMINE? SLEPC_DEFAULT_TOL/10.0: svd->tol;
    if (maxit==PETSC_DETERMINE) maxit = svd->max_it;
    PetscCall(EPSSetTolerances(cyclic->eps,tol,maxit));
    switch (svd->conv) {
    case SVD_CONV_ABS:
      PetscCall(EPSSetConvergenceTest(cyclic->eps,EPS_CONV_ABS));break;
    case SVD_CONV_REL:
      PetscCall(EPSSetConvergenceTest(cyclic->eps,EPS_CONV_REL));break;
    case SVD_CONV_NORM:
      if (svd->isgeneralized) {
        if (!svd->nrma) PetscCall(MatNorm(svd->OP,NORM_INFINITY,&svd->nrma));
        if (!svd->nrmb) PetscCall(MatNorm(svd->OPb,NORM_INFINITY,&svd->nrmb));
        PetscCall(EPSSetConvergenceTestFunction(cyclic->eps,EPSConv_Cyclic,svd,NULL));
      } else {
        PetscCall(EPSSetConvergenceTest(cyclic->eps,EPS_CONV_NORM));break;
      }
      break;
    case SVD_CONV_MAXIT:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Maxit convergence test not supported in this solver");
    case SVD_CONV_USER:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"User-defined convergence test not supported in this solver");
    }
  }
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
  /* Transfer the trackall option from svd to eps */
  PetscCall(SVDGetTrackAll(svd,&trackall));
  PetscCall(EPSSetTrackAll(cyclic->eps,trackall));
  /* Transfer the initial subspace from svd to eps */
  if (svd->nini<0 || svd->ninil<0) {
    for (i=0;i<-PetscMin(svd->nini,svd->ninil);i++) {
      PetscCall(MatCreateVecs(cyclic->C,&v,NULL));
      PetscCall(VecGetArrayWrite(v,&va));
      if (svd->isgeneralized) PetscCall(MatGetLocalSize(svd->B,&p,NULL));
      k = (svd->isgeneralized && svd->which==SVD_SMALLEST)? p: m;  /* size of upper block row */
      if (i<-svd->ninil) {
        PetscCall(VecGetArrayRead(svd->ISL[i],&isa));
        if (svd->isgeneralized) {
          PetscCall(VecGetLocalSize(svd->ISL[i],&isl));
          PetscCheck(isl==m+p,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for left initial vector");
          offset = (svd->which==SVD_SMALLEST)? m: 0;
          PetscCall(PetscArraycpy(va,isa+offset,k));
        } else {
          PetscCall(VecGetLocalSize(svd->ISL[i],&isl));
          PetscCheck(isl==k,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for left initial vector");
          PetscCall(PetscArraycpy(va,isa,k));
        }
        PetscCall(VecRestoreArrayRead(svd->IS[i],&isa));
      } else PetscCall(PetscArrayzero(&va,k));
      if (i<-svd->nini) {
        PetscCall(VecGetLocalSize(svd->IS[i],&isl));
        PetscCheck(isl==n,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Size mismatch for right initial vector");
        PetscCall(VecGetArrayRead(svd->IS[i],&isa));
        PetscCall(PetscArraycpy(va+k,isa,n));
        PetscCall(VecRestoreArrayRead(svd->IS[i],&isa));
      } else PetscCall(PetscArrayzero(va+k,n));
      PetscCall(VecRestoreArrayWrite(v,&va));
      PetscCall(VecDestroy(&svd->IS[i]));
      svd->IS[i] = v;
    }
    svd->nini = PetscMin(svd->nini,svd->ninil);
    PetscCall(EPSSetInitialSpace(cyclic->eps,-svd->nini,svd->IS));
    PetscCall(SlepcBasisDestroy_Private(&svd->nini,&svd->IS));
    PetscCall(SlepcBasisDestroy_Private(&svd->ninil,&svd->ISL));
  }
  PetscCall(EPSSetUp(cyclic->eps));
  PetscCall(EPSGetDimensions(cyclic->eps,NULL,&svd->ncv,&svd->mpd));
  svd->ncv = PetscMin(svd->ncv,PetscMin(M,N));
  PetscCall(EPSGetTolerances(cyclic->eps,NULL,&svd->max_it));
  if (svd->tol==(PetscReal)PETSC_DETERMINE) svd->tol = SLEPC_DEFAULT_TOL;

  svd->leftbasis = PETSC_TRUE;
  PetscCall(SVDAllocateSolution(svd,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### ```SVDCyclicGetECrossMat```

#### Что делает функция?

Составляет расширенную матрицу кросс произведения, которая используется в алгоритмах сингулярного разложения (SVD) в циклическом методе.

#### Параметры:  
SVD svd - ???  
Mat A - Матрица для создания блока $A^T*A$  
Mat AT - транспонированная матрица для создания блока $A^T*A$  
Mat *C - получаемая расширенная матрица кросс произведения (extended cross product matrix) 
Vec t - вспомогательный вектор, нужный для определения размеров единичной матрицы $I_m$​ - и для $M$ и для $m$.

#### Возвращаемое значение
PetscErrorCode - коды ошибок

#### Детали имплементации

1. Первые строки - инициализация переменных
```
  SVD_CYCLIC       *cyclic = (SVD_CYCLIC*)svd->data;
  SVD_CYCLIC_SHELL *ctx;
  PetscInt         i,M,N,m,n,Istart,Iend;
  VecType          vtype;
  Mat              Id,Zm,Zn,ATA;
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  PetscBool        gpu;
  const PetscInt   *ranges;
  PetscMPIInt      size;
#endif
```
ctx — контекст для работы с неявной-матрицей.  
i — индекс ???  
M, N — глобальные размеры матрицы 𝐴.  
m, n — локальные размеры матрицы 𝐴 для текущего процесса.  
Istart, Iend — начало и конец диапазона индексов для текущего процесса.  
vtype — тип векторов, используемых в матрице.  
Zm, Zn — вспомогательные матрицы для создания циклической матрицы.  
ATA - временная матрица для хранения $A^TA$  

Последующие несколько строк заполняют переменные.

2. Далее определяется, является ли матрица явной
```
  if (cyclic->explicitmatrix) {
```
Если условие истинно, то делается проверка
```
    PetscCheck(svd->expltrans,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Cannot use explicit cyclic matrix with implicit transpose");
```

Проверяет, что при исп явной матрицы C также используется явный метод обратной проверки или вычислений для транспонированной матрицы A 
(svd->expltrans = PETSC_TRUE)

далее идет построение C для явной матрицы.   
Перемножение обычной и транспонированной матрицы
```
    PetscCall(MatAssemblyEnd(Zn,MAT_FINAL_ASSEMBLY));
    PetscCall(MatProductCreate(AT,A,NULL,&ATA));
```
Создание блочной матрицы  
```
MatCreateTile(1.0, Id, 1.0, Zm, 1.0, Zn, 1.0, ATA, C);
```

3. Если матрица неявная, то 

Создание контекста и вспомогательных векторов???    

Создание неявной матрицы
```
PetscCall(MatCreateShell(PetscObjectComm((PetscObject)svd),m+n,m+n,M+N,M+N,ctx,C));
```

Определение "операций на матрице"
```
    PetscCall(MatShellSetOperation(*C,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_ECross));
    PetscCall(MatShellSetOperation(*C,MATOP_DESTROY,(void(*)(void))MatDestroy_ECross));
```
где 

MATOP_GET_DIAGONAL: функция для извлечения диагональных элементов.
MATOP_DESTROY: функция для очистки памяти.
MATOP_MULT: умножение матрицы на вектор

Этот блок занимается созданием оболочечной (shell) матрицы для представления C, используя контекст и функции, вместо явного хранения всех её элементов.


Тут что то страшное с выравниванием (непроверенная инфа)
```
ctx->misaligned = (((ranges[i+1] - ranges[i]) * sizeof(PetscScalar)) % 16) ? PETSC_TRUE : PETSC_FALSE;
```

# References.

1. SLEPc Users Manual Scalable Library for Eigenvalue Problem Computations, Carmen Campos Jose E. Roman, Eloy Romero Andres Tomas. Раздел EPS: Eigenvalue Problem Solver, страница 17.  
2. http://www.grycap.upv.es/slepc.  
3. https://petsc.org/release/  
4. https://slepc.upv.es/slepc-main/include/slepc/private/
5. https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition