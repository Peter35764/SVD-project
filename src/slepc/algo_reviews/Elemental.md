## ВВЕДЕНИЕ

Этот алгоритм представляет собой реализацию обёртки библиотеки **Elemental** в SLEPC. Для полного ознакомления с возможностями и специфическими особенностями библиотеки **Elemental** предлагается рассмотреть [References 1](https://github.com/elemental/Elemental).

Отметим важное:
- Библиотека **Elemental** реализована на языке C++ и позволяет пользователю решать большие задачи, например, находить сингулярное разложение большой матрицы, путём распараллеливания.
- Эта библиотека обладает не самыми устаревшими, но весьма гибкими способами распараллеливания.
- **Elemental**, в сущности, реализует методы сингулярного разложения, представленные в LAPACK.
- С помощью этой библиотеки можно реализовать rsvd(??). Эта возможность появляется по причине того, что в **Elemental** имеется возможность генерировать случайные матрицы! Как правило, случайные матрицы нужны для проверки алгоритмов, но, выходит, что на основе **Elemental** можно реализовать rsvd.
- В данный момент библиотека не поддерживается, но имеет своё ответвление в виде библиотеки **Hydrogen** ([References 7](https://github.com/LLNL/Elemental)).

>**Note**: Поскольку концептуально обёртки библиотек ScaLAPACK и **Elemental** в SLEPC имеют одно и то же предназначение в виде распараллеливания больших задач и решения «параллелей» как обычных задач, стоит отметить, что существование этих обёрток скорее имеет практическую значимость, ибо идейно они не имеют принципиальных различий — за исключением методов распараллеливания.

>**Note2**: Практическая значимость выражается в том, что из-за разных методов распараллеливания эти обёртки подойдут для разных ситуаций.

## ИМПЛЕМЕНТАЦИЯ

Поскольку обёртка **Elemental** реализует методы из LAPACK, то концептуально имплементация выглядит так:
1. Осуществляется «перевод» данных о задаче из «формата» SLEPC в «формат» **Elemental**.
2. Находится сингулярное разложение с использованием методов LAPACK, интегрированных в библиотеку **Elemental**.
3. Осуществляется «перевод» найденных результатов в «формат» SLEPC.

## REFERENCES
1. [https://github.com/elemental/Elemental](https://github.com/elemental/Elemental) — репозиторий библиотеки **Elemental**.
2. [https://gitlab.com/slepc/slepc/-/blob/main/src/svd/impls/external/elemental/svdelemen.cxx?ref_type=heads](https://gitlab.com/slepc/slepc/-/blob/main/src/svd/impls/external/elemental/svdelemen.cxx?ref_type=heads) — исходный код библиотеки **Elemental** для SLEPC.
3. [https://elemental.github.io/documentation/dev/index.html](https://elemental.github.io/documentation/dev/index.html) — Документация для разработчиков.
4. [https://gitlab.com/petsc/petsc/-/blob/main/include/petsc/private/petscelemental.h](https://gitlab.com/petsc/petsc/-/blob/main/include/petsc/private/petscelemental.h) — тут представлены методы, осуществляющие «перевод форматов».
5. [https://elemental.github.io/documentation/dev/lapack_like/spectral/SVD.html](https://elemental.github.io/documentation/dev/lapack_like/spectral/SVD.html) — страница документации библиотеки **Elemental**, из которой видно, что в этой библиотеке представлены известные нам алгоритмы сингулярного разложения.
6. [https://github.com/elemental/Elemental/blob/master/src/lapack_like/spectral/SVD.cpp](https://github.com/elemental/Elemental/blob/master/src/lapack_like/spectral/SVD.cpp) — файл, реализующий «решатель SVD» в **Elemental**.
7. [https://github.com/LLNL/Elemental](https://github.com/LLNL/Elemental) — потомок **Elemental**.
