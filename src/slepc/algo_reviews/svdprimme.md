# Введение 

Этот файл представляет собой обёртку для библиотеки PRIMME.\
PRIMME - это высокопроизводительная библиотека для вычисления некоторого количества собственных значений/собственных векторов, а также сингулярных значений/векторов. PRIMME особенно хорошо оптимизирована для больших и сложных задач. Поддерживаются действительные симметричные и комплексные эрмитовы задачи, как стандартные A x = λ x, так и обобщённые A x = λ B x.

## Имплиментация 

файл svdprimme.c представляет собой обёртку для библиотки primme 
### ```multMatvec_PRIMME```
```c++
static void multMatvec_PRIMME(void *xa,PRIMME_INT *ldx,void *ya,PRIMME_INT *ldy,int *blockSize,int *transpose,struct primme_svds_params *primme,int *ierr)
{
  PetscInt   i;
  SVD_PRIMME *ops = (SVD_PRIMME*)primme->matrix;
  Vec        x = ops->x,y = ops->y;
  SVD        svd = ops->svd;

  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    if (*transpose) {
      PetscCallAbort(PetscObjectComm((PetscObject)svd),VecPlaceArray(y,(PetscScalar*)xa+(*ldx)*i));
      PetscCallAbort(PetscObjectComm((PetscObject)svd),VecPlaceArray(x,(PetscScalar*)ya+(*ldy)*i));
      PetscCallAbort(PetscObjectComm((PetscObject)svd),MatMult(svd->AT,y,x));
    } else {
      PetscCallAbort(PetscObjectComm((PetscObject)svd),VecPlaceArray(x,(PetscScalar*)xa+(*ldx)*i));
      PetscCallAbort(PetscObjectComm((PetscObject)svd),VecPlaceArray(y,(PetscScalar*)ya+(*ldy)*i));
      PetscCallAbort(PetscObjectComm((PetscObject)svd),MatMult(svd->A,x,y));
    }
    PetscCallAbort(PetscObjectComm((PetscObject)svd),VecResetArray(x));
    PetscCallAbort(PetscObjectComm((PetscObject)svd),VecResetArray(y));
  }
  PetscFunctionReturnVoid();
}
```
Это функция релизует умножение матрицы(или транспонированной матрицы) на вектор.
По сути получает входные массивы векторов xa,ya и их размеры.
В зависимости от параметра transpose вызывает либо умножение на матрицу, либо на транспонированную матрицу.
Входные данные:\
xa: массив чисел, в котором лежат входные векторы подряд.\
ldx: размер шага по массиву xa для доступа к элементам каждого вектора.\
blockSize: количество одновременно обрабатываемых векторов.\
transpose: флаг, указывающий, умножать ли на транспонированную матрицу.\
primme: структура с информацией о матрице и векторах, необходимой для вычислений.\
Выходные данные:\
ya: массив чисел, в который записываются результаты умножения матрицы (или её транспонированной версии) на входные векторы.\
ldy: размер шага по массиву ya, определяющий, как размещать результаты.\

### ```par_GlobalSumReal```
```c++
static void par_GlobalSumReal(void *sendBuf,void *recvBuf,int *count,primme_svds_params *primme,int *ierr)
{
  if (sendBuf == recvBuf) {
    *ierr = MPI_Allreduce(MPI_IN_PLACE,recvBuf,*count,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)primme->commInfo));
  } else {
    *ierr = MPI_Allreduce(sendBuf,recvBuf,*count,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)primme->commInfo));
  }
}
```
Выполняет глобальное суммирование выщественных значений между процессами.
Использует MPI_Allreduce длля суммированиия значений из всех процессов. Если входные и выходные данные совпадают используется оптимизированный режим MPI_IN_PLACE.\
Входные данные:
void *sendBuf - указатель на буффер, содержащий данные, которые нужно суммировать.\
void *recvBuf - указатель на  буффер, куда будут записаны результаты суммирования.\
int *count - указатель на счётчик количества элементов в массиве sendBuf.\
primme_svds_params *primme - указатель на структуру параметров PRIMME SVDS.\
int *ierr - указатель для хранения кода ошибки.\
Выходные данные:\
Статус операци(Успешное завершение или код ошибки MPI).\
Глобальную сумму, которая содержится в recvBuf.\
### ```SVDSetUp_PRIMME```
Функция `SVDSetUp_PRIMME` настраивает вычисления сингулярных значений и векторов (SVD) с использованием библиотеки PRIMME. Она включает этапы проверки, инициализации параметров PRIMME, переноса настроек из SLEPc и выделения рабочего пространства.

```c++
PetscMPIInt        numProcs, procID;
SVD_PRIMME         *ops = (SVD_PRIMME*)svd->data;
primme_svds_params *primme = &ops->primme;

PetscFunctionBegin;
SVDCheckStandard(svd);
SVDCheckDefinite(svd);
PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)svd), &numProcs));
PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)svd), &procID));
```
Извлекаются данные о параллельной среде:
numProcs: Количество процессов.
procID: Идентификатор текущего процесса.
Проверяется корректность вида задачи SVD (например, стандартный или обобщенный).

```c++
PetscCall(MatGetSize(svd->A, &m, &n));
PetscCall(MatGetLocalSize(svd->A, &mloc, &nloc));
PetscCall(SVDSetDimensions_Default(svd));
if (svd->max_it == PETSC_DETERMINE) svd->max_it = PETSC_INT_MAX;
svd->leftbasis = PETSC_TRUE;
SVDCheckUnsupported(svd, SVD_FEATURE_STOPPING);
```
Получаются размеры матрицы svd->A:
m, n: Глобальные размеры матрицы.
mloc, nloc: Локальные размеры матрицы.
Устанавливаются значения по умолчанию:
Максимальное число итераций (max_it).
Флаг необходимости вычисления левой базы (leftbasis).

```c++
primme_svds_free(primme);
  primme_svds_initialize(primme);
  primme->m             = (PRIMME_INT)m;
  primme->n             = (PRIMME_INT)n;
  primme->mLocal        = (PRIMME_INT)mloc;
  primme->nLocal        = (PRIMME_INT)nloc;
  primme->numSvals      = (int)svd->nsv;
  primme->matrix        = ops;
  primme->commInfo      = svd;
  primme->maxMatvecs    = (PRIMME_INT)svd->max_it;
#if !defined(SLEPC_HAVE_PRIMME2p2)
  primme->eps           = SlepcDefaultTol(svd->tol);
#endif
  primme->numProcs      = numProcs;
  primme->procID        = procID;
  primme->printLevel    = 1;
  primme->matrixMatvec  = multMatvec_PRIMME;
  primme->globalSumReal = par_GlobalSumReal;
#if defined(SLEPC_HAVE_PRIMME3)
  primme->broadcastReal = par_broadcastReal;
#endif
#if defined(SLEPC_HAVE_PRIMME2p2)
  primme->convTestFun   = convTestFun;
  primme->monitorFun    = monitorFun;
#endif
  if (ops->bs > 0) primme->maxBlockSize = (int)ops->bs;

  switch (svd->which) {
    case SVD_LARGEST:
      primme->target = primme_svds_largest;
      break;
    case SVD_SMALLEST:
      primme->target = primme_svds_smallest;
      break;
  }
```
Тут инициализируются параметры Primme

```c++
switch (svd->which) {
  case SVD_LARGEST:
    primme->target = primme_svds_largest;
    break;
  case SVD_SMALLEST:
    primme->target = primme_svds_smallest;
    break;
}
```
Определяется метод поиска:\
SVD_LARGEST: Поиск наибольших сингулярных значений.\
SVD_SMALLEST: Поиск наименьших сингулярных значений.

```c++
if (svd->mpd != PETSC_DETERMINE) {
  primme->maxBasisSize = (int)svd->mpd;
  if (svd->ncv != PETSC_DETERMINE) 
    PetscCall(PetscInfo(svd, "Warning: 'ncv' is ignored by PRIMME\n"));
} else if (svd->ncv != PETSC_DETERMINE) {
  primme->maxBasisSize = (int)svd->ncv;
}
```
Обработка пользовательских настроек
mpd: Максимальное количество направлений поиска.
ncv: Размер пространства Крылова.
При конфликте настроек выводится предупреждение.


### ```SVDSolve_PRIMME```

Функция для решения задачи.

```c++
SVD_PRIMME *ops = (SVD_PRIMME*)svd->data;
PetscScalar *svecs, *a;
PetscInt i,ierrprimme,ld;
PetscReal *svals,*rnorms;
```
Инициализируются необходимые переменные, указатели и структуры для выполнения задачи.
ops — указатель на структуру данных PRIMME, содержащую параметры алгоритма.
svecs и a — указатели для хранения векторов сингулярного разложения.
svals и rnorms — массивы для хранения сингулярных значений и ошибок норм.

```c++
ops->primme.aNorm = 0.0;
ops->primme.initSize = (int)svd->nini;
ops->primme.iseed[0] = -1;
ops->primme.iseed[1] = -1;
ops->primme.iseed[2] = -1;
ops->primme.iseed[3] = -1;
```
Сбрасываются некоторые параметры PRIMME для подготовки к новому запуску.
aNorm — начальная норма матрицы.
initSize — начальный размер подпространства.
iseed — начальное состояние для генерации случайных чисел.
```c++
PetscCall(PetscCalloc1(ops->primme.numSvals*(ops->primme.mLocal+ops->primme.nLocal),&svecs));
PetscCall(PetscMalloc2(svd->ncv,&svals,svd->ncv,&rnorms));
ierrprimme = PRIMME_DRIVER(svals,svecs,rnorms,&ops->primme);
for (i=0;i<svd->ncv;i++) svd->sigma[i] = svals[i];
for (i=0;i<svd->ncv;i++) svd->errest[i] = rnorms[i];
PetscCall(PetscFree2(svals,rnorms));
svd->nconv = ops->primme.initSize >= 0 ? (PetscInt)ops->primme.initSize : 0;
svd->reason = svd->nconv >= svd->nsv ? SVD_CONVERGED_TOL : SVD_DIVERGED_ITS;
PetscCall(PetscIntCast(ops->primme.stats.numOuterIterations,&svd->its));
```
Тут решается основная задача, копируются результаты и устанавливается статус решения.

Примечание:
Основная функция PRIMME для решения задачи SVD. В зависимости от конфигурации она может использовать один из следующих вариантов:
1. cprimme_svds (комплексные числа, одинарная точность)
2. zprimme_svds (комплексные числа, двойная точность)
3. sprimme_svds (вещественные числа, одинарная точность)
4. dprimme_svds (вещественные числа, двойная точность)
```c++
if (ierrprimme != 0) {
  switch (ierrprimme%100) {
    case -1:
      SETERRQ(..., "unexpected error");
    case -2:
      SETERRQ(..., "allocation error");
    case -3:
      break;
    default:
      PetscCheck(..., "configuration error; check PRIMME's manual");
      PetscCheck(..., "runtime error; check PRIMME's manual");
  }
}
```
Обаботка ошибок.
### ```SVDReset_PRIMME```
```c++
static PetscErrorCode SVDReset_PRIMME(SVD svd)
{
  SVD_PRIMME     *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  primme_svds_free(&ops->primme);
  PetscCall(VecDestroy(&ops->x));
  PetscCall(VecDestroy(&ops->y));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
Эта функция освобождает ресурсы, связанные с PRIMME, которые были выделены в процессе выполнения. Она используется для сброса состояния объекта перед его повторным использованием или уничтожением.
Как правило вызывается перед уничтожением объекта и перед повторной инициализации объекта для решения другой задачи.

### ```SVDSetFromOptions_PRIMME```
```c++
static PetscErrorCode SVDSetFromOptions_PRIMME(SVD svd,PetscOptionItems *PetscOptionsObject)
{
  SVD_PRIMME      *ctx = (SVD_PRIMME*)svd->data;
  PetscInt        bs;
  SVDPRIMMEMethod meth;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SVD PRIMME Options");

    PetscCall(PetscOptionsInt("-svd_primme_blocksize","Maximum block size","SVDPRIMMESetBlockSize",ctx->bs,&bs,&flg));
    if (flg) PetscCall(SVDPRIMMESetBlockSize(svd,bs));

    PetscCall(PetscOptionsEnum("-svd_primme_method","Method for solving the singular value problem","SVDPRIMMESetMethod",SVDPRIMMEMethods,(PetscEnum)ctx->method,(PetscEnum*)&meth,&flg));
    if (flg) PetscCall(SVDPRIMMESetMethod(svd,meth));

  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

```
Эта функция считывает параметры, заданные пользователем, из командной строки или из конфигурационных файлов PETSc/SLEPc и настраивает решатель SVD PRIMME в соответствии с этими параметрами.
PetscOptions считывает параметры из командной строки, в данном случае размер блока и выбранный метод для решения.

### ```SVDPRIMMESetMethod```

```c++
PetscErrorCode SVDPRIMMESetMethod(SVD svd,SVDPRIMMEMethod method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svd,method,2);
  PetscTryMethod(svd,"SVDPRIMMESetMethod_C",(SVD,SVDPRIMMEMethod),(svd,method));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
Какие методы можно передавать? Методы PRIMME:
1. **`SVD_PRIMME_HYBRID`**
    **Описание:** Гибридный метод, сочетающий несколько подходов для достижения высокой производительности.
2. **`SVD_PRIMME_DYNAMIC`**
    **Описание:** Метод с динамическим выбором стратегии на основе текущих вычислений.
3. **`SVD_PRIMME_DEFAULT_MIN_TIME`**
    **Описание:** Метод, минимизирующий время выполнения.
4. **`SVD_PRIMME_LOWER`**
    **Описание:** Метод для поиска наименьших сингулярных значений.
5. **`SVD_PRIMME_UPPER`**
    **Описание:** Метод для поиска наибольших сингулярных значений.
6. **`SVD_PRIMME_JDQR`**
    **Описание:** Метод Джекоби-Дэвидсона (Jacobi-Davidson) для задач на сингулярные значения.
7. **`SVD_PRIMME_GD`**
    **Описание:** Градиентный метод.
8. **`SVD_PRIMME_GD_PLUSK`**
    **Описание:** Модифицированный градиентный метод с улучшением через дополнительные базисные векторы.
9. **`SVD_PRIMME_GD_Olsen_PLUSK`**
    **Описание:** Вариация метода `GD_PLUSK`, оптимизированная для некоторых типов матриц.
10. **`SVD_PRIMME_DEFAULT_MIN_RESIDUAL`**
    **Описание:** Метод, минимизирующий резидуал (остаточную ошибку).

### ```SVDPRIMMEGetMethod_PRIMME```
```c++
static PetscErrorCode SVDPRIMMEGetMethod_PRIMME(SVD svd,SVDPRIMMEMethod *method)
{
  SVD_PRIMME *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  *method = (SVDPRIMMEMethod)ops->method;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
Функция  возвращает текущий метод решения, установленный для задачи SVD с использованием библиотеки PRIMME. Этот метод позволяет пользователю узнать, какой предопределенный метод PRIMME используется для решения задачи.

### ```SVDPRIMMEGetMethod ```
```c++
PetscErrorCode SVDPRIMMEGetMethod(SVD svd,SVDPRIMMEMethod *method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscAssertPointer(method,2);
  PetscUseMethod(svd,"SVDPRIMMEGetMethod_C",(SVD,SVDPRIMMEMethod*),(svd,method));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
Функция предназначена для получения текущего метода, используемого библиотекой PRIMME для вычисления сингулярных значений. Метод возвращается пользователю через указатель на переменную, позволяя узнать настройки текущего метода вычислений.

### ``` SVDCreate_PRIMME ```
```c++
SLEPC_EXTERN PetscErrorCode SVDCreate_PRIMME(SVD svd)
{
  SVD_PRIMME     *primme;

  PetscFunctionBegin;
  PetscCall(PetscNew(&primme));
  svd->data = (void*)primme;

  primme_svds_initialize(&primme->primme);
  primme->bs = 0;
  primme->method = (primme_svds_preset_method)SVD_PRIMME_HYBRID;
  primme->svd = svd;

  svd->ops->solve          = SVDSolve_PRIMME;
  svd->ops->setup          = SVDSetUp_PRIMME;
  svd->ops->setfromoptions = SVDSetFromOptions_PRIMME;
  svd->ops->destroy        = SVDDestroy_PRIMME;
  svd->ops->reset          = SVDReset_PRIMME;
  svd->ops->view           = SVDView_PRIMME;

  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMESetBlockSize_C",SVDPRIMMESetBlockSize_PRIMME));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMEGetBlockSize_C",SVDPRIMMEGetBlockSize_PRIMME));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMESetMethod_C",SVDPRIMMESetMethod_PRIMME));
  PetscCall(PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMEGetMethod_C",SVDPRIMMEGetMethod_PRIMME));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
Функция `SVDCreate_PRIMME` создает и инициализирует контекст PRIMME для решения задачи вычисления сингулярных значений (SVD). Она выделяет память, настраивает параметры PRIMME по умолчанию, и связывает соответствующие функции-обработчики для различных операций, таких как решение, настройка, очистка и отображение.

---

# Referenses
[Документация слепка](https://slepc.upv.es/documentation/)\
[github PRIMME](https://github.com/primme/primme)\
