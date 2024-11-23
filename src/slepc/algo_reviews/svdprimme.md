# Введение 

Этот файл представляет собой обёртку для библиотеки PRIMME

# Имплиментация 

файл svdprimme.c представляет собой обёртку для библиотки primme 
## multMatvec_PRIMME
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
## par_GlobalSumReal
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
Использует MPI_Allreduce длля суммированиия значений из всех процессов. Если входные и выходные данные совпадают используется оптимизированный режим MPI_IN_PLACE
## SVDSetUp_PRIMME
```c++
static PetscErrorCode SVDSetUp_PRIMME(SVD svd)
{
  PetscMPIInt        numProcs,procID;
  PetscInt           n,m,nloc,mloc;
  SVD_PRIMME         *ops = (SVD_PRIMME*)svd->data;
  primme_svds_params *primme = &ops->primme;

  PetscFunctionBegin;
  SVDCheckStandard(svd);
  SVDCheckDefinite(svd);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)svd),&numProcs));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)svd),&procID));

  /* Check some constraints and set some default values */
  PetscCall(MatGetSize(svd->A,&m,&n));
  PetscCall(MatGetLocalSize(svd->A,&mloc,&nloc));
  PetscCall(SVDSetDimensions_Default(svd));
  if (svd->max_it==PETSC_DETERMINE) svd->max_it = PETSC_INT_MAX;
  svd->leftbasis = PETSC_TRUE;
  SVDCheckUnsupported(svd,SVD_FEATURE_STOPPING);
#if !defined(SLEPC_HAVE_PRIMME2p2)
  if (svd->converged != SVDConvergedAbsolute) PetscCall(PetscInfo(svd,"Warning: using absolute convergence test\n"));
#endif

  /* Transfer SLEPc options to PRIMME options */
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

  /* If user sets mpd or ncv, maxBasisSize is modified */
  if (svd->mpd!=PETSC_DETERMINE) {
    primme->maxBasisSize = (int)svd->mpd;
    if (svd->ncv!=PETSC_DETERMINE) PetscCall(PetscInfo(svd,"Warning: 'ncv' is ignored by PRIMME\n"));
  } else if (svd->ncv!=PETSC_DETERMINE) primme->maxBasisSize = (int)svd->ncv;

  PetscCheck(primme_svds_set_method(ops->method,(primme_preset_method)EPS_PRIMME_DEFAULT_MIN_TIME,PRIMME_DEFAULT_METHOD,primme)>=0,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"PRIMME method not valid");

  svd->mpd = (PetscInt)primme->maxBasisSize;
  svd->ncv = (PetscInt)(primme->locking?svd->nsv:0)+primme->maxBasisSize;
  ops->bs  = (PetscInt)primme->maxBlockSize;

  /* Set workspace */
  PetscCall(SVDAllocateSolution(svd,0));

  /* Prepare auxiliary vectors */
  if (!ops->x) PetscCall(MatCreateVecsEmpty(svd->A,&ops->x,&ops->y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

```
Настройка контекста SVD для использования PRIMME.
тут происходит проверка корректности матриц, инициализация структуры параметров PRIMME, передача параметров SLEPc в PRIMME и выделение памяти.

## SVDSolve_PRIMME

Функция для решения задачи.

```c++
static PetscErrorCode SVDSolve_PRIMME(SVD svd)
{
  SVD_PRIMME     *ops = (SVD_PRIMME*)svd->data;
  PetscScalar    *svecs, *a;
  PetscInt       i,ierrprimme,ld;
  PetscReal      *svals,*rnorms;

  PetscFunctionBegin;
  /* Reset some parameters left from previous runs */
  ops->primme.aNorm    = 0.0;
  ops->primme.initSize = (int)svd->nini;
  ops->primme.iseed[0] = -1;
  ops->primme.iseed[1] = -1;
  ops->primme.iseed[2] = -1;
  ops->primme.iseed[3] = -1;

  /* Allocating left and right singular vectors contiguously */
  PetscCall(PetscCalloc1(ops->primme.numSvals*(ops->primme.mLocal+ops->primme.nLocal),&svecs));

  /* Call PRIMME solver */
  PetscCall(PetscMalloc2(svd->ncv,&svals,svd->ncv,&rnorms));
  ierrprimme = PRIMME_DRIVER(svals,svecs,rnorms,&ops->primme);
  for (i=0;i<svd->ncv;i++) svd->sigma[i] = svals[i];
  for (i=0;i<svd->ncv;i++) svd->errest[i] = rnorms[i];
  PetscCall(PetscFree2(svals,rnorms));

  /* Copy left and right singular vectors into svd */
  PetscCall(BVGetLeadingDimension(svd->U,&ld));
  PetscCall(BVGetArray(svd->U,&a));
  for (i=0;i<ops->primme.initSize;i++) PetscCall(PetscArraycpy(a+i*ld,svecs+i*ops->primme.mLocal,ops->primme.mLocal));
  PetscCall(BVRestoreArray(svd->U,&a));

  PetscCall(BVGetLeadingDimension(svd->V,&ld));
  PetscCall(BVGetArray(svd->V,&a));
  for (i=0;i<ops->primme.initSize;i++) PetscCall(PetscArraycpy(a+i*ld,svecs+ops->primme.mLocal*ops->primme.initSize+i*ops->primme.nLocal,ops->primme.nLocal));
  PetscCall(BVRestoreArray(svd->V,&a));

  PetscCall(PetscFree(svecs));

  svd->nconv  = ops->primme.initSize >= 0 ? (PetscInt)ops->primme.initSize : 0;
  svd->reason = svd->nconv >= svd->nsv ? SVD_CONVERGED_TOL: SVD_DIVERGED_ITS;
  PetscCall(PetscIntCast(ops->primme.stats.numOuterIterations,&svd->its));

  /* Process PRIMME error code */
  if (ierrprimme != 0) {
    switch (ierrprimme%100) {
      case -1:
        SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"PRIMME library failed with error code=%" PetscInt_FMT ": unexpected error",ierrprimme);
      case -2:
        SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"PRIMME library failed with error code=%" PetscInt_FMT ": allocation error",ierrprimme);
      case -3: /* stop due to maximum number of iterations or matvecs */
        break;
      default:
        PetscCheck(ierrprimme<-39,PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"PRIMME library failed with error code=%" PetscInt_FMT ": configuration error; check PRIMME's manual",ierrprimme);
        PetscCheck(ierrprimme>=-39,PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"PRIMME library failed with error code=%" PetscInt_FMT ": runtime error; check PRIMME's manual",ierrprimme);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
```
Сначала инициализируются параметры и выделяется память, затем вызывается функция из PRIMME для решения задачи свд, тут PRIMME_DRIVER - макрос, который зависит от конфигурации, например вещественный или комплексный.
Затем копируются вычесленные сингулярные значения, левые и правые сингулярные векторы и нормы ошибок, после этого устанавливается статус вычеслений и обрабатываются ошибки PRIMME.

Примечание:
Основная функция PRIMME для решения задачи SVD. В зависимости от конфигурации она может использовать один из следующих вариантов:
1. cprimme_svds (комплексные числа, одинарная точность)
2. zprimme_svds (комплексные числа, двойная точность)
3. sprimme_svds (вещественные числа, одинарная точность)
4. dprimme_svds (вещественные числа, двойная точность)

## SVDReset_PRIMME
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

## SVDSetFromOptions_PRIMME
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

## SVDPRIMMESetMethod

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
1. **SVD_PRIMME_HYBRID**
    **Описание:** Гибридный метод, сочетающий несколько подходов для достижения высокой производительности.
2. **SVD_PRIMME_DYNAMIC**
    **Описание:** Метод с динамическим выбором стратегии на основе текущих вычислений.
3. **SVD_PRIMME_DEFAULT_MIN_TIME**
    **Описание:** Метод, минимизирующий время выполнения.
4. **SVD_PRIMME_LOWER**
    **Описание:** Метод для поиска наименьших сингулярных значений.
5. **SVD_PRIMME_UPPER**
    **Описание:** Метод для поиска наибольших сингулярных значений.
6. **SVD_PRIMME_JDQR**
    **Описание:** Метод Джекоби-Дэвидсона (Jacobi-Davidson) для задач на сингулярные значения.
7. **SVD_PRIMME_GD**
    **Описание:** Градиентный метод.
8. **`SVD_PRIMME_GD_PLUSK`**
    **Описание:** Модифицированный градиентный метод с улучшением через дополнительные базисные векторы.
9. **`SVD_PRIMME_GD_Olsen_PLUSK`**
    **Описание:** Вариация метода `GD_PLUSK`, оптимизированная для некоторых типов матриц.
10. **`SVD_PRIMME_DEFAULT_MIN_RESIDUAL`**
    **Описание:** Метод, минимизирующий резидуал (остаточную ошибку).

---

# Referenses
[Документация слепка](https://slepc.upv.es/documentation/)
[Документация PRIMME](https://github.com/primme/primme)