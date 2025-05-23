{
  description = "SMAP super-resolution project environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
      {
        devShells = forAllSystems (system:
          let
            pkgs = import nixpkgs {
              inherit system;
              overlays = [
                (final: prev: {
                  python3Packages = prev.python3Packages.overrideScope (self: super: {
                    lineax = super.lineax.overridePythonAttrs (old: rec {
                      doCheck = false;
                    });
                    jax-cuda12-pjrt = super.jax-cuda12-pjrt.overridePythonAttrs (old: rec {
                      src = prev.fetchPypi {
                        version = "0.6.1";
                        platform = "manylinux2014_x86_64";
                        dist = "py3";
                        hash = "sha256-TJfRClqawJ+gAVaMrDtxUBTo27ws2GdjdT9Y5acwwzM=";
                        pname = "jax_cuda12_pjrt";
                        format = "wheel";
                        python = "py3";
                      };
                    });
                    jax-cuda12-plugin = super.jax-cuda12-plugin.overridePythonAttrs (old: rec {
                      version = "0.6.1";
                      src = prev.fetchPypi {
                        inherit version;
                        platform = "manylinux2014_x86_64";
                        dist = "cp312";
                        hash = "sha256-GIXxW+OPrszPvySxhP/cHQ02Nxfq3SU01XWcDT0K9SM=";
                        pname = "jax_cuda12_plugin";
                        format = "wheel";
                        abi = "cp312";
                        python = "cp312";
                      };
                    });
                    jaxlib = super.jaxlib.overridePythonAttrs (old: rec {
                      version = "0.6.1";
                      src = prev.fetchPypi {
                        inherit version;
                        platform = "manylinux2014_x86_64";
                        dist = "cp312";
                        hash = "sha256-0DkSRGhWW785NjsVBMGQ5nGeaviaeUje4lbx3ugTu5Q=";
                        pname = "jaxlib";
                        format = "wheel";
                        abi = "cp312";
                        python = "cp312";
                      };
                    });
                    jax = super.jax.overridePythonAttrs (old: rec {
                      version = "0.6.1";
                      src =  prev.fetchFromGitHub {
                        owner = "google";
                        repo = "jax";
                        rev = "382506f1705db9c9ac348b9783497e310feef6a5";
                        hash = "sha256-Am+ksPC4U3vL5LGmePzSaMSwWJOCcVrC+DFkJzJP+1U=";
                      };
                      doCheck = false;
                      doInstallCheck = false;
                    });
                  });
                })
              ];
              config = {
                allowBroken = false;
                allowUnfree = true;
                cudaSupport = true;
              }; };
            diffrax = pkgs.python3Packages.buildPythonPackage rec {
              pname = "diffrax";
              version = "0.7.0";
              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "sha256-87zFeM2SqcqG/G9aVMHedsG6YvdN5ptWACYkvyBTFvE=";
              };
              pyproject = true;
              propagatedBuildInputs = with pkgs.python3Packages; [
                jax
                jaxlib
                equinox
                typing-extensions
                hatchling
                lineax
                optimistix
                numpy
              ];
              doCheck = false;
            };
            pyresample = pkgs.python3Packages.buildPythonPackage rec {
              pname = "pyresample";
              version = "1.34.1";
              src = pkgs.fetchPypi {
                inherit pname version;
                sha256 = "sha256-bg58zwkLyr9L/IgY1/9jN0H935eE/pC0rdhvkltbcvE=";
              };
              pyproject = true;
              propagatedBuildInputs = with pkgs.python3Packages; [
                numpy
                pykdtree
                shapely
                configobj
                donfig
                platformdirs
                pyproj
                setuptools
                versioneer
                scipy
                pillow
                dask
                xarray
                cython
              ];
              doCheck = false;
            };
          in {
            default = pkgs.mkShell {
              name = "smapsr";
              LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
              packages = [
                pkgs.python3Packages.jaxlib
                pkgs.python3Packages.jax
                pkgs.python3Packages.equinox
                pkgs.python3Packages.optax
                pkgs.python3Packages.einops
                pkgs.python3Packages.netcdf4
                pkgs.python3Packages.h5py
                pkgs.python3Packages.xarray
                pkgs.python3Packages.matplotlib
                pkgs.python3Packages.ipython
                pkgs.python3Packages.geopandas
                pkgs.python3Packages.rasterio
                pkgs.python3Packages.rioxarray
                pkgs.python3Packages.scikit-image
                pkgs.python3Packages.jupyter
                pkgs.python3Packages.pyqt6
                pkgs.python3Packages.tqdm
                pkgs.cudatoolkit
                pkgs.cudaPackages.cudnn
                pkgs.linuxPackages.nvidia_x11
                pkgs.python3Packages.pyflakes
                pkgs.python3Packages.jupytext
                pkgs.python3Packages.python-lsp-server
                pkgs.python3Packages.isort
                diffrax
                pyresample
              ];
            };
          });
      };
}
