{ pkgs, lib, config, inputs, ... }:

{

  name = "smapsr";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.zlib
    pkgs.gmt
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.linuxPackages.nvidia_x11
    pkgs.python3Packages.matplotlib
  ];

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
      };
    };
  };

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/basics/
  enterShell = ''
    . .devenv/state/venv/bin/activate
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
