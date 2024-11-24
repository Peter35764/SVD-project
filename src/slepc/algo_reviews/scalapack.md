# ВВЕДЕНИЕ

Так как этот алгоритм в SLEPC — по сути, свёртка из библиотеки ScaLAPACK, то для общих сведений кратко опишем саму библиотеку ScaLAPACK, исходя из руководства пользователя (см. [References[1]](https://netlib.org/scalapack/slug/node1.html#SECTION01000000000000000000)):

- **ScaLAPACK** — это библиотека высокопроизводительных процедур линейной алгебры для компьютеров MIMD с распределённой памятью и сетями рабочих станций, поддерживающих PVM и/или MPI.
- Компоненты программного обеспечения ScaLAPACK: LAPACK, BLAS, PBLAS, BLACS, MPI.

> **NOTE:** Компоненты бывают разные (см. [References[4]](https://netlib.org/scalapack/slug/node11.html#SECTION04130000000000000000)).

### LAPACK
- ScaLAPACK — это продолжение проекта LAPACK, результатом которого является разработка аналогичного программного обеспечения для рабочих станций, векторных суперкомпьютеров и параллельных компьютеров с общей памятью. Таким образом, решена задача масштабирования LAPACK на большие архитектуры компьютеров.
- **Схожести ScaLAPACK и LAPACK:**
  - Обе библиотеки содержат процедуры для решения систем линейных уравнений, задач наименьших квадратов и задач на собственные значения.
  - Целями обоих проектов (ScaLAPACK и LAPACK) являются: (см. [References[2]](https://netlib.org/scalapack/slug/node9.html#SECTION04110000000000000000))

> **NOTE:** LAPACK будет работать на любой машине, где доступны BLAS, а ScaLAPACK — на любой машине, где доступны и BLAS, и BLACS.  
> **NOTE2:** Процедуры в библиотеках хоть и схожи, но не идентичны.

### BLAS — Обеспечивает работу LAPACK
- Это стандарт, используемый в данном контексте для LAPACK. В BLAS крайне оптимизированы основные операции линейной алгебры, и LAPACK, как более крупный пакет, пользуется качественными оптимизациями из BLAS.
- Важной целью BLAS является обеспечение уровня переносимости для вычислений.

### PBLAS — Обеспечивает параллелизм
- Это реализация BLAS, обеспечивающая вычислительную основу для ScaLAPACK (см. [References[6]](https://netlib.org/scalapack/slug/node14.html#SECTION04133000000000000000)).

### BLACS — Обеспечивает общение между процессами
- Это библиотека передачи сообщений, разработанная для линейной алгебры.
- Простое умозаключение по [References[7]](https://netlib.org/scalapack/slug/node15.html#SECTION04134000000000000000): эта библиотека обеспечивает общение параллельных процессов — процессы «начинают общаться на языке линейной алгебры как на родном».

# ОПИСАНИЕ АЛГОРИТМА

По сути, алгоритм представляет собой реализацию свёртки ScaLAPACK.

Перед описанием алгоритма, давайте разберём, какой именно алгоритм реализован. Для этого опишем путь, состоящий из подключённых библиотек:

- **Svdscalap.c** — код имплементации ScaLAPACK в SLEPC ([References[8]](https://gitlab.com/slepc/slepc/-/blob/main/src/svd/impls/external/scalapack/svdscalap.c?ref_type=heads)).
- **Slepcscalapack.h** — заголовочный файл, согласовывает имена функций из PETSC в SLEPC ([References[9]](https://gitlab.com/slepc/slepc/-/blob/main/include/slepc/private/slepcscalapack.h)).
- **Petscscalapack.h** — заголовочный файл, согласовывает имена функций из ScaLAPACK в PETSC ([References[10]](https://gitlab.com/petsc/petsc/-/blob/main/include/petsc/private/petscscalapack.h)).
> Именно здесь стыкуются описанные в введении библиотеки, подробную реализацию коих мы опускаем.
- **Petscblaslapack.h** — заголовочный файл, согласовывает имена функций из LAPACK и BLAS в PETSC ([References[11]](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack.h)).
  > Нас интересует 286-я строка.
- **Petscblaslapack_mangle.h** — заголовочный файл, также согласовывает имена функций LAPACK и BLAS в PETSC ([References[12]](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack_mangle.h)).
  > Нас интересует 169-я строка.

Делаем промежуточное умозаключение — нужно найти репозиторий ScaLAPACK с функцией `gesvd`. См. описание алгоритмов от Коли.

# ИМПЛЕМЕНТАЦИЯ

Перед началом этого замечательного приключения хотелось бы написать следующее предупреждение:

Этот обзор не затрагивает вопросы касательно полной программисткой имплементации, поскольку рассмотрение всех нюансов работы с памятью и подключаемыми библиотеками превратит обзор в «книгу» по ScaLAPACK — а целью является именно обзор. 

Засим заключим джентльменское соглашение, что все возникшие вопросы подобного порядка, по типу «как конкретно реализована работа с библиотеками PBLAS, BLACS?» — оставляются на откуп читателю и рассматриваются в [REFERENCES](#references).

Сначала рассмотрим, как имплементация выглядит в SLEPC.  
> **Тут я описываю процедурные вызовы вплоть до соприкосновения с самим алгоритмом, взятым из ScaLAPACK.**

# REFERENCES

1. [User’s guide for Scalapack](https://netlib.org/scalapack/slug/node1.html#SECTION01000000000000000000)
2. [User’s guide: Scalapack](https://netlib.org/scalapack/slug/node9.html#SECTION04110000000000000000)
3. [Репозиторий ScaLAPACK](https://github.com/Reference-ScaLAPACK/scalapack)
4. [ScaLAPACK Software components](https://netlib.org/scalapack/slug/node11.html#SECTION04130000000000000000)
5. [Страница LAPACK в репозитории netlib](https://www.netlib.org/lapack/#_users_guide)
6. [PBLAS](https://netlib.org/scalapack/slug/node14.html#SECTION04133000000000000000)
7. [BLACS (Basic Linear Algebra Communication Subprograms)](https://netlib.org/scalapack/slug/node15.html#SECTION04134000000000000000)
8. [Код имплементации ScaLAPACK в SLEPC](https://gitlab.com/slepc/slepc/-/blob/main/src/svd/impls/external/scalapack/svdscalap.c?ref_type=heads)
9. [Заголовочный файл, согласовывающий имена функций из PETSC в SLEPC](https://gitlab.com/slepc/slepc/-/blob/main/include/slepc/private/slepcscalapack.h)
10. [Заголовочный файл, согласовывающий имена функций из ScaLAPACK в PETSC](https://gitlab.com/petsc/petsc/-/blob/main/include/petsc/private/petscscalapack.h)
11. [Заголовочный файл, согласовывающий имена функций из LAPACK и BLAS в PETSC](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack.h)
12. [Заголовочный файл, согласовывающий имена функций из LAPACK и BLAS в PETSC – для non-Microsoft Windows systems](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack_mangle.h)
13. [Реализация математических операций в PETSC посредством BLAS и LAPACK](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack.h)
