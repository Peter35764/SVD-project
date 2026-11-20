{
  description = "SVD-project nix flake";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = {nixpkgs, ...}: let
    forEachSystem = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;
  in {
    devShells = forEachSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      pythonEnv = pkgs.python3.withPackages (ps: [
        ps.matplotlib
        ps.pyqt6
      ]);
    in {
      default = pkgs.mkShell.override {stdenv = pkgs.llvmPackages.stdenv;} {
        nativeBuildInputs = [
          pkgs.cmake
          pkgs.gfortran
          pkgs.ninja
          pkgs.pkg-config
          pkgs.clang-tools

          pythonEnv

          (pkgs.writeShellScriptBin "build" ''cmake -B build -G "Ninja" && cmake --build build'')
        ];
        buildInputs = [
          pkgs.lapack
          pkgs.eigen
          pkgs.boost
        ];
      };
    });
  };
}
