with import <nixpkgs> { };
let
  pythonEnv = python39.withPackages (ps: [
    ps.virtualenvwrapper
    ps.pip
  ]);
in
mkShell {
  name = "Glaucoma";
  buildInputs = [
    pythonEnv
    yarn
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    nodejs
    stdenv
    libpqxx
    zlib
    zlib.dev
    libffi
    libffi.dev
  ];
  shellHook = ''
    # Allow the use of wheels.
    SOURCE_DATE_EPOCH=$(date +%s)
    # Augment the dynamic linker path
    VENV=venv
    if test ! -d $VENV; then
      virtualenv $VENV
    fi
    source ./$VENV/bin/activate
    export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
    export LD_LIBRARY_PATH=${lib.makeLibraryPath [ glib stdenv.cc.cc.lib cudaPackages.cudatoolkit
    xorg.libX11 freeglut libGLU libGL linuxPackages.nvidia_x11 oracle-instantclient zlib
    zlib.dev cudaPackages.cudnn]}
    export LD_LIBRARY_PATH=$(printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
  '';
}
