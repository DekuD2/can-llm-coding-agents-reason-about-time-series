{
  description = "Development flake for the paper 'Can LLM Coding Agents Reason About Time Series?'";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python311
              pkgs.uv
            ];

            env = lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH =
              let
                # libpackages = pkgs.pythonManylinuxPackages.manylinux1 ++ [
                libpackages = [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                ];
              in
                (lib.makeLibraryPath libpackages) + ":/run/opengl-driver/lib";
            };

            # Increase timeout because nvidia packages are very large.
            UV_HTTP_TIMOEOUT = 5000;

            shellHook = ''
              unset PYTHONPATH
              uv sync
              source .venv/bin/activate
            '';
          };
        }
      );
    };
}

