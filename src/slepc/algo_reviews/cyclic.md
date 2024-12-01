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


## Имплементация.

### Функция MatMult_Cyclic(Mat B,Vec x,Vec y) - функция которая принимает на вход примает входную матрицу, входной вектор, выходной вектор. Позволяет умножить матрицу B на вектор x с помощью стандартных циклических операций.

Входные данные: 
- Матрица B;
- Вектор x;
- Вектор y.\
Выходные данные:
- Вектор y= B*x.\

```
static PetscErrorCode MatMult_Cyclic(Mat B,Vec x,Vec y)
{
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          m;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(MatGetLocalSize(ctx->A,&m,NULL));
  PetscCall(VecGetArrayRead(x,&px));
  PetscCall(VecGetArrayWrite(y,&py));
  PetscCall(VecPlaceArray(ctx->x1,px));
  PetscCall(VecPlaceArray(ctx->x2,px+m));
  PetscCall(VecPlaceArray(ctx->y1,py));
  PetscCall(VecPlaceArray(ctx->y2,py+m));
  PetscCall(MatMult(ctx->A,ctx->x2,ctx->y1));
  PetscCall(MatMult(ctx->AT,ctx->x1,ctx->y2));
  PetscCall(VecResetArray(ctx->x1));
  PetscCall(VecResetArray(ctx->x2));
  PetscCall(VecResetArray(ctx->y1));
  PetscCall(VecResetArray(ctx->y2));
  PetscCall(VecRestoreArrayRead(x,&px));
  PetscCall(VecRestoreArrayWrite(y,&py));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция MatGetDiagonal_Cyclic(Mat B,Vec diag) - функция, которая возвращает нулевой вектор.

Входные данные:
- Матрица B;
- Вектор diag.\
Выходные данные:
- Нулевой вектор diag, сопоставимый с размерами матрица B.\


```
static PetscErrorCode MatGetDiagonal_Cyclic(Mat B,Vec diag)
{
  PetscFunctionBegin;
  PetscCall(VecSet(diag,0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция MatDestroy_Cyclic(Mat B) - деструктор матрица B.

Входные данные:
- Матрица B.
Выходные данные: -.
```
static PetscErrorCode MatDestroy_Cyclic(Mat B)
{
  SVD_CYCLIC_SHELL *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(VecDestroy(&ctx->x1));
  PetscCall(VecDestroy(&ctx->x2));
  PetscCall(VecDestroy(&ctx->y1));
  PetscCall(VecDestroy(&ctx->y2));
  if (ctx->misaligned) {
    PetscCall(VecDestroy(&ctx->wx2));
    PetscCall(VecDestroy(&ctx->wy2));
  }
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция MatMult_ECross(Mat B,Vec x,Vec y) - функция которая принимает на вход примает входную матрицу, входной вектор, выходной вектор. Позволяет умножить кросс-матрицу B на вектор x для вычислений с дополнительными ограничениями или временными значениями.

Входные данные: 
- Матрица B;
- Вектор x;
- Вектор y.\
Выходные данные:
- Вектор y= B*x.\
```
static PetscErrorCode MatMult_ECross(Mat B,Vec x,Vec y)
{
  SVD_CYCLIC_SHELL  *ctx;
  const PetscScalar *px;
  PetscScalar       *py;
  PetscInt          mn,m,n;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(MatGetLocalSize(ctx->A,NULL,&n));
  PetscCall(VecGetLocalSize(y,&mn));
  m = mn-n;
  PetscCall(VecGetArrayRead(x,&px));
  PetscCall(VecGetArrayWrite(y,&py));
  PetscCall(VecPlaceArray(ctx->x1,px));
  PetscCall(VecPlaceArray(ctx->x2,px+m));
  PetscCall(VecPlaceArray(ctx->y1,py));
  PetscCall(VecPlaceArray(ctx->y2,py+m));
  PetscCall(VecCopy(ctx->x1,ctx->y1));
  PetscCall(MatMult(ctx->A,ctx->x2,ctx->w));
  PetscCall(MatMult(ctx->AT,ctx->w,ctx->y2));
  PetscCall(VecResetArray(ctx->x1));
  PetscCall(VecResetArray(ctx->x2));
  PetscCall(VecResetArray(ctx->y1));
  PetscCall(VecResetArray(ctx->y2));
  PetscCall(VecRestoreArrayRead(x,&px));
  PetscCall(VecRestoreArrayWrite(y,&py));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция MatDestroy_ECross(Mat B) - деструктор кросс-матрицы B.\
Входные данные: 
- Кросс-матрица B.\
Выходные данные: -. 
```
static PetscErrorCode MatDestroy_ECross(Mat B)
{
  SVD_CYCLIC_SHELL *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(B,&ctx));
  PetscCall(VecDestroy(&ctx->x1));
  PetscCall(VecDestroy(&ctx->x2));
  PetscCall(VecDestroy(&ctx->y1));
  PetscCall(VecDestroy(&ctx->y2));
  PetscCall(VecDestroy(&ctx->diag));
  PetscCall(VecDestroy(&ctx->w));
  if (ctx->misaligned) {
    PetscCall(VecDestroy(&ctx->wx2));
    PetscCall(VecDestroy(&ctx->wy2));
  }
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция EPSConv_Cyclic(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx) - функция, которая обеспечивает корректную оценку ошибки, определяет сходимости алгоритма.\

Входные данные:
- EPS eps - представляет контекст задачи нахождения собственных значений.
- PetscScalar eigr хранит реальную часть текущего вычисленного собственного значения.
- PetscScalar eigi хранит мнимую часть текущего вычисленного собственного значения.
- PetscReal res ошибка для найденного собственного значения.
- PetscReal *errest указывает, куда функция должна записать вычисленную оценку ошибки.
- void *ctx передается объект типа SVD, содержащий параметры сингулярного разложения матриц.\
Выходные данные:
- PETSC_SUCCESS: Функция завершилась успешно, иначе код ошибки.
- PetscReal errest вычисленная оценка ошибки сходимости.
  
```
static PetscErrorCode EPSConv_Cyclic(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  SVD svd = (SVD)ctx;

  PetscFunctionBegin;
  *errest = res/PetscMax(svd->nrma,svd->nrmb);
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция  SVDSolve_Cyclic(SVD svd) - функция, которая реализует алгоритм решения задачи сингулярного разложения с использованием циклического метода.\

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.\
Выходные данные:
- обновленная структура SVD: svd->sigma: Массив найденных сингулярных значений; svd->nconv: Количество найденных сингулярных значений; svd->its: Количество итераций, потребовавшихся для решения; svd->reason: Причина завершения вычислений.\

1. Объявление локальных переменных.
2. Вызывает функцию EPSSolve, которая решает задачу нахождения собственных значений, используя объект EPS.
3. Определяет, сколько собственных значений сошлось, и сохраняет результат в nconv.
4. Извлекает количество итераций, выполненных для решения задачи, и сохраняет его в поле svd->its.
5. Получает причину завершения вычислений и записывает ее в svd->reason.
6. Цикл для обработки всех сошедшихся собственных значений:
- Получает i-е собственное значение: er — реальная часть, ei — мнимая часть.
- Проверяет корректность λ=er+ei⋅i (собственного значения) для задачи сингулярного разложения. преобразуется в сингулярное значение σ.
- Проверка: используется только положительное σ, так как сингулярные значения по определению неотрицательны.
- Если задача является обобщенной (svd->isgeneralized) и требуется минимальное сингулярное значение (svd->which == SVD_SMALLEST), используется обратное значение $1/σ$.
- В остальных случаях сингулярное значение не меняется.
7.Сохраняет количество найденных сингулярных значений в svd->nconv.

```
static PetscErrorCode SVDSolve_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscInt       i,j,nconv;
  PetscScalar    er,ei;
  PetscReal      sigma;

  PetscFunctionBegin;
  PetscCall(EPSSolve(cyclic->eps));
  PetscCall(EPSGetConverged(cyclic->eps,&nconv));
  PetscCall(EPSGetIterationNumber(cyclic->eps,&svd->its));
  PetscCall(EPSGetConvergedReason(cyclic->eps,(EPSConvergedReason*)&svd->reason));
  for (i=0,j=0;i<nconv;i++) {
    PetscCall(EPSGetEigenvalue(cyclic->eps,i,&er,&ei));
    PetscCall(SVDCyclicCheckEigenvalue(svd,er,ei,&sigma,NULL));
    if (sigma>0.0) {
      if (svd->isgeneralized && svd->which==SVD_SMALLEST) svd->sigma[j] = 1.0/sigma;
      else svd->sigma[j] = sigma;
      j++;
    }
  }
  svd->nconv = j;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция SVDComputeVectors_Cyclic(SVD svd) - выбирает подходящую реализацию вычисления сингулярных векторов, исходя из типа задачи, и вызывает соответствующую специализированную функцию.\

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.\
Выходные данные:
- обновленная структура SVD.
```
static PetscErrorCode SVDComputeVectors_Cyclic(SVD svd)
{
  PetscFunctionBegin;
  switch (svd->problem_type) {
    case SVD_STANDARD:
      PetscCall(SVDComputeVectors_Cyclic_Standard(svd));
      break;
    case SVD_GENERALIZED:
      PetscCall(SVDComputeVectors_Cyclic_Generalized(svd));
      break;
    case SVD_HYPERBOLIC:
      PetscCall(SVDComputeVectors_Cyclic_Hyperbolic(svd));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_WRONG,"Unknown singular value problem type");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Функция EPSMonitor_Cyclic(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx) - реализует мониторинг процесса нахождения собственных значений в циклическом методе, связанного с задачей сингулярного разложения (SVD). Она выполняет обновление текущих сингулярных значений и ошибок их оценки, преобразуя собственные значения в сингулярные.

Входные данные:
- EPS eps - представляет контекст задачи нахождения собственных значений.
- PetscScalar eigr хранит реальную часть текущего вычисленного собственного значения.
- PetscScalar eigi хранит мнимую часть текущего вычисленного собственного значения.
- PetscReal res ошибка для найденного собственного значения.
- PetscReal *errest указывает, куда функция должна записать вычисленную оценку ошибки.
- void *ctx передается объект типа SVD, содержащий параметры сингулярного разложения матриц.\
Выходные данные:
- PETSC_SUCCESS: Функция завершилась успешно, иначе код ошибки.
- PetscReal errest вычисленная оценка ошибки сходимости.

```
static PetscErrorCode EPSMonitor_Cyclic(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i,j;
  SVD            svd = (SVD)ctx;
  PetscScalar    er,ei;
  PetscReal      sigma;
  ST             st;

  PetscFunctionBegin;
  nconv = 0;
  PetscCall(EPSGetST(eps,&st));
  for (i=0,j=0;i<PetscMin(nest,svd->ncv);i++) {
    er = eigr[i]; ei = eigi[i];
    PetscCall(STBackTransform(st,1,&er,&ei));
    PetscCall(SVDCyclicCheckEigenvalue(svd,er,ei,&sigma,NULL));
    if (sigma>0.0) {
      svd->sigma[j]  = sigma;
      svd->errest[j] = errest[i];
      if (errest[i] && errest[i] < svd->tol) nconv++;
      j++;
    }
  }
  nest = j;
  PetscCall(SVDMonitor(svd,its,nconv,svd->sigma,svd->errest,nest));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDSetFromOptions_Cyclic(SVD svd,PetscOptionItems *PetscOptionsObject) - предназначена для настройки параметров задачи сингулярного разложения (SVD) методом cyclic на основе пользовательских опций. Она: Читает параметры, заданные через командную строку или программный интерфейс. Настраивает объект SVD в соответствии с этими параметрами. Обеспечивает корректное поведение метода с учетом выбранных настроек.\

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- PetscOptionItems *PetscOptionsObject передает пользовательские параметры.
Выходные данные:
- обновленная структура SVD.

```
static PetscErrorCode SVDSetFromOptions_Cyclic(SVD svd,PetscOptionItems *PetscOptionsObject)
{
  PetscBool      set,val;
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  ST             st;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SVD Cyclic Options");

    PetscCall(PetscOptionsBool("-svd_cyclic_explicitmatrix","Use cyclic explicit matrix","SVDCyclicSetExplicitMatrix",cyclic->explicitmatrix,&val,&set));
    if (set) PetscCall(SVDCyclicSetExplicitMatrix(svd,val));

  PetscOptionsHeadEnd();

  if (!cyclic->eps) PetscCall(SVDCyclicGetEPS(svd,&cyclic->eps));
  if (!cyclic->explicitmatrix && !cyclic->usereps) {
    /* use as default an ST with shell matrix and Jacobi */
    PetscCall(EPSGetST(cyclic->eps,&st));
    PetscCall(STSetMatMode(st,ST_MATMODE_SHELL));
  }
  PetscCall(EPSSetFromOptions(cyclic->eps));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDCyclicSetExplicitMatrix_Cyclic(SVD svd,PetscBool explicitmat) - обновляет параметр использования явной матрицы в методе циклического сингулярного разложения (SVD).\ 

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- PetscBool explicitmat - логический параметр, указывающий, использовать ли явную матрицу в вычислениях.\
Выходные данные:
- PetscBool explicitmat.
  
```
static PetscErrorCode SVDCyclicSetExplicitMatrix_Cyclic(SVD svd,PetscBool explicitmat)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (cyclic->explicitmatrix != explicitmat) {
    cyclic->explicitmatrix = explicitmat;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

Функция SVDCyclicSetExplicitMatrix(SVD svd,PetscBool explicitmat) - обновляет параметр использования явной матрицы сингулярного разложения (SVD).\ 

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- PetscBool explicitmat - логический параметр, указывающий, использовать ли явную матрицу в вычислениях.\
Выходные данные:
- PetscBool explicitmat.

```
PetscErrorCode SVDCyclicSetExplicitMatrix(SVD svd,PetscBool explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,explicitmat,2);
  PetscTryMethod(svd,"SVDCyclicSetExplicitMatrix_C",(SVD,PetscBool),(svd,explicitmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDCyclicGetExplicitMatrix_Cyclic(SVD svd,PetscBool *explicitmat) - возвращает информацию о том, используется ли явная матрица в расчетах.\

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- указатель на PetscBool explicitmat - логический параметр, указывающий, использовать ли явную матрицу в вычислениях.\
Выходные данные:
- PetscBool explicitmat.

```
static PetscErrorCode SVDCyclicGetExplicitMatrix_Cyclic(SVD svd,PetscBool *explicitmat)
{
  SVD_CYCLIC *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  *explicitmat = cyclic->explicitmatrix;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDCyclicSetEPS(SVD svd,EPS eps) - используется для настройки объекта EPS (eigensolver) в контексте задачи сингулярного разложения (SVD) методом cyclic. Объект EPS является решателем собственных значений, который используется внутри метода cyclic для вычислений.\
Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- объект EPS, который содержит данные для собственных значений. \
Выходные данные: -.

```
PetscErrorCode SVDCyclicSetEPS(SVD svd,EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(svd,1,eps,2);
  PetscTryMethod(svd,"SVDCyclicSetEPS_C",(SVD,EPS),(svd,eps));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
### Фукнция SVDCyclicSetEPS(SVD svd,EPS eps) - используется для получения объекта решателя собственных значений (EPS) в контексте метода cyclic для сингулярного разложения (SVD). Если объект EPS еще не создан, функция выполняет его инициализацию и настройку.\
Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- Указатель на переменную, в которую будет записан объект EPS. \
Выходные данные:
- Указатель на объект EPS.

```
static PetscErrorCode SVDCyclicGetEPS_Cyclic(SVD svd,EPS *eps)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  if (!cyclic->eps) {
    PetscCall(EPSCreate(PetscObjectComm((PetscObject)svd),&cyclic->eps));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)cyclic->eps,(PetscObject)svd,1));
    PetscCall(EPSSetOptionsPrefix(cyclic->eps,((PetscObject)svd)->prefix));
    PetscCall(EPSAppendOptionsPrefix(cyclic->eps,"svd_cyclic_"));
    PetscCall(PetscObjectSetOptions((PetscObject)cyclic->eps,((PetscObject)svd)->options));
    PetscCall(EPSSetWhichEigenpairs(cyclic->eps,EPS_LARGEST_REAL));
    PetscCall(EPSMonitorSet(cyclic->eps,EPSMonitor_Cyclic,svd,NULL));
  }
  *eps = cyclic->eps;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDCyclicGetEPS(SVD svd,EPS *eps) - вызывает метод, специфичный для текущего типа SVD.\
Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- Указатель на переменную, в которую будет записан объект EPS. \
Выходные данные:-.\

```
PetscErrorCode SVDCyclicGetEPS(SVD svd,EPS *eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscAssertPointer(eps,2);
  PetscUseMethod(svd,"SVDCyclicGetEPS_C",(SVD,EPS*),(svd,eps));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDView_Cyclic(SVD svd,PetscViewer viewer) - предназначена для отображения информации о состоянии метода cyclic для задачи сингулярного разложения (SVD). Она выводит информацию о настройках метода в указанный PetscViewer. \
Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.
- Объект PPetscViewer для вывода информации. \
Выходные данные:
- PetscViewer viewer.\

```
static PetscErrorCode SVDView_Cyclic(SVD svd,PetscViewer viewer)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (!cyclic->eps) PetscCall(SVDCyclicGetEPS(svd,&cyclic->eps));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %s matrix\n",cyclic->explicitmatrix?"explicit":"implicit"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(EPSView(cyclic->eps,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция SVDReset_Cyclic(SVD svd) - используется для сброса состояния метода cyclic в задаче сингулярного разложения (SVD). Она освобождает ресурсы, связанные с этим методом, включая объект EPS и вспомогательные матрицы. \
Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.\
Выходные данные:-\
```
static PetscErrorCode SVDReset_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  PetscCall(EPSReset(cyclic->eps));
  PetscCall(MatDestroy(&cyclic->C));
  PetscCall(MatDestroy(&cyclic->D));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция  SVDDestroy_Cyclic(SVD svd) - освобождает все ресурсы, связанные с методом cyclic для задачи сингулярного разложения (SVD). Она уничтожает объект EPS, освобождает память, используемую структурой данных SVD_CYCLIC, и удаляет связанные с методом cyclic функции. \
Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.\
Выходные данные:-\

```
static PetscErrorCode SVDDestroy_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic = (SVD_CYCLIC*)svd->data;

  PetscFunctionBegin;
  PetscCall(EPSDestroy(&cyclic->eps));
  PetscCall(PetscFree(svd->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Фукнция  SVDCreate_Cyclic - создает и инициализирует объект для задачи сингулярного разложения (SVD) с использованием метода cyclic. Она настраивает структуру данных SVD и задает функции, специфичные для метода cyclic.. \

Входные данные:
- объект SVD (сингулярное разложение), который содержит данные и параметры задачи.\
Выходные данные:
- Обновленные объект SVD.

```
SLEPC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD svd)
{
  SVD_CYCLIC     *cyclic;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cyclic));
  svd->data                = (void*)cyclic;
  svd->ops->solve          = SVDSolve_Cyclic;
  svd->ops->solveg         = SVDSolve_Cyclic;
  svd->ops->solveh         = SVDSolve_Cyclic;
  svd->ops->setup          = SVDSetUp_Cyclic;
  svd->ops->setfromoptions = SVDSetFromOptions_Cyclic;
  svd->ops->destroy        = SVDDestroy_Cyclic;
  svd->ops->reset          = SVDReset_Cyclic;
  svd->ops->view           = SVDView_Cyclic;
  svd->ops->computevectors = SVDComputeVectors_Cyclic;
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetEPS_C",SVDCyclicSetEPS_Cyclic));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetEPS_C",SVDCyclicGetEPS_Cyclic));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicSetExplicitMatrix_C",SVDCyclicSetExplicitMatrix_Cyclic));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDCyclicGetExplicitMatrix_C",SVDCyclicGetExplicitMatrix_Cyclic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

# References.

1. SLEPc Users Manual Scalable Library for Eigenvalue Problem Computations, Carmen Campos Jose E. Roman, Eloy Romero Andres Tomas. Раздел EPS: Eigenvalue Problem Solver, страница 17.  
2. http://www.grycap.upv.es/slepc.  
3. https://petsc.org/release/  
4. https://slepc.upv.es/slepc-main/include/slepc/private/
5. https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition
