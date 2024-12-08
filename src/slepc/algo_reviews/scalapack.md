## ВВЕДЕНИЕ

Этот алгоритм в SLEPC представляет собой обёртку библиотеки ScaLAPACK. Подробная информация об библиотеке представлена в [References](https://netlib.org/scalapack/slug/node1.html#SECTION01000000000000000000).

Что важно знать, так это:

- **ScaLAPACK** – библиотека, с помощью которой можно распараллелить большие задачи, и для каждой из «параллелей» применять тот или иной метод из **LAPACK**, например, методы сингулярного разложения.
- В данный момент библиотека **ScaLAPACK** считается устаревшей в смысле технологий распараллеливания.

## ИМПЛЕМЕНТАЦИЯ

Резюмируем:

В сущности, обёртка **ScaLAPACK** реализует методы **SVD** из библиотеки **LAPACK**.

Сама имплементация этой обёртки в SLEPC делает следующее:

1. «Переводит» данные о задаче из структур и объектов SLEPC в «формат» ScaLAPACK.
2. Находит сингулярное разложение.
3. Переводит найденные результаты в «формат» SLEPC.

## REFERENCES

1. [User’s guide for Scalapack](https://netlib.org/scalapack/slug/node1.html#SECTION01000000000000000000)
2. [User’s guide: Scalapack](https://netlib.org/scalapack/slug/node9.html#SECTION04110000000000000000)
3. [Репозиторий ScaLAPACK](https://github.com/Reference-ScaLAPACK/scalapack)
4. [ScaLAPACK Software components](https://netlib.org/scalapack/slug/node11.html#SECTION04130000000000000000)
5. [Страница LAPACK в репозитории netlib](https://www.netlib.org/lapack/#_users_guide)
6. [PBLAS](https://netlib.org/scalapack/slug/node14.html#SECTION04133000000000000000)
7. [BLACS (Basic Linear Algebra Communication Subprograms)](https://netlib.org/scalapack/slug/node15.html#SECTION04134000000000000000)
8. [Код имплементации ScaLAPACK в SLEPC](https://gitlab.com/slepc/slepc/-/blob/main/src/svd/impls/external/scalapack/svdscalap.c?ref_type=heads)
9. [Заголовочный файл, согласовывает имена функций из PETSC в SLEPC](https://gitlab.com/slepc/slepc/-/blob/main/include/slepc/private/slepcscalapack.h)
10. [Заголовочный файл, согласовывает имена функций из ScaLAPACK в PETSC](https://gitlab.com/petsc/petsc/-/blob/main/include/petsc/private/petscscalapack.h)
11. [Заголовочный файл, согласовывает имена функций из LAPACK и BLAS в PETSC](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack.h)
12. [Заголовочный файл, согласовывает имена функций из LAPACK и BLAS в PETSC для non-Microsoft Windows systems](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack_mangle.h)
13. [Реализация математических операций в PETSC посредством BLAS и LAPACK](https://gitlab.com/petsc/petsc/-/blob/main/include/petscblaslapack.h)
