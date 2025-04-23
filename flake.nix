{
  description = "Flake for SVD project";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };
  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {pkgs, ...}: let
        llvmPkgs = pkgs.llvmPackages;
        lib = pkgs.lib;
        libraries = [
          pkgs.lapack
          llvmPkgs.libcxx
          llvmPkgs.libllvm
          llvmPkgs.lldb
        ];

        pythonEnv = pkgs.python3.withPackages (ps:
          with ps; [
            pyqt6
          ]);
      in {
        devShells.default = pkgs.mkShell.override {stdenv = llvmPkgs.stdenv;} {
          nativeBuildInputs = [
            llvmPkgs.clang
            pkgs.pkg-config
            pkgs.cmakeCurses
            (pkgs.writeShellScriptBin "build" ''cmake -G "Ninja" CMakeLists.txt && ninja'') # build project via CMake
            (pkgs.writeShellScriptBin "run" ''cmake -G "Ninja" CMakeLists.txt && ninja && ./svd_test'') # run project via CMake
            (pkgs.writeShellScriptBin "bd" ''cmake -G "Ninja" CMakeLists.txt && ninja'') # alias for build
            (pkgs.writeShellScriptBin "compare" ''python ./src/benchmarks/compare_csv.py'')
            (pkgs.writeShellScriptBin "dv" ''nix develop path:$PWD'')
          ];
          buildInputs = with pkgs;
            libraries
            ++ [
              cmake
              eigen
              boost
              llvmPkgs.clang-tools
              ninja
              pythonEnv
            ];
          LD_LIBRARY_PATH = "${llvmPkgs.stdenv.cc.cc.lib.outPath}/lib:${pkgs.lib.makeLibraryPath libraries}";
          CLANGD_FLAGS = "--query-driver=${llvmPkgs.clang}/bin/clang";
          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" [pkgs.lapack]}";
          CPATH = builtins.concatStringsSep ":" [
            (lib.makeSearchPath "include/eigen3" [pkgs.eigen])
            (lib.makeSearchPath "resource-root/include" [llvmPkgs.clang])
            (lib.makeSearchPathOutput "dev" "include" [llvmPkgs.libcxx pkgs.lapack])
          ];
        };
      };
    };
}
