echo Setting up environment for TSF
echo

# cuda
echo Please install cuda on your own from the NVIDIA website
echo

# Install bazel #######################
read -r -p "Install Bazel [y/N] " response
case "$response" in
  [yY][eY][eS]|[yY])
    echo installing bazel

    sudo apt-get install openjdk-8-jdk

    grep "14.04" /etc/issue > /dev/null && (sudo add-apt-repository ppa:webupd8team/javai && sudo apt-get update && sudo apt-get install oracle-java8-installer)

    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
    sudo apt-get update && sudo apt-get install bazel
    ;;
  *)
    echo Not Installing Bazel
esac
echo
#######################################


# Install qt4 ########################
read -r -p "Install Qt4 [y/N] " response
case "$response" in
  [yY][eY][eS]|[yY])
    echo Installing Qt4
    sudo apt-get install libqt4-dev
    ;;
  *)
    echo Not Installing Qt4
esac
echo
#######################################

# OpenSceneGraph
read -r -p "Build OpenSceneGraph [y/N] " response
case "$response" in
  [yY][eY][eS]|[yY])
    echo Building OpenSceneGraph

    cd OpenSceneGraph
    mkdir build -p
    cd build
    cmake -D CMAKE_C_FLAGS="-fPIC" \
          -D CMAKE_CXX_FLAGS="-fPIC" \
          -D CMAKE_INSTALL_PREFIX="local_install" \
          -D BUILD_OSG_APPLICATIONS="OFF" \
          -D FFMPEG_LIBAVCODEC_INCLUDE_DIRS="" \
          ../

    make -j $(nproc)
    make install
    cd ../..
    ;;
  *)
    echo Not building OpenSceneGraph
esac
echo
#######################################

# Get data
read -r -p "Get Data [y/N] " response
case "$response" in
  [yY][eY][eS]|[yY])
    read -p "Enter a path for storing the data: " -i "$HOME" -e path

    mkdir -p $path
    cd $path

    echo Downloading data

    if wget http://robots.engin.umich.edu/~aushani/tsf_data.tar.gz
    then
      echo Got tarball
    else
      echo Couldn\'t connect to http://robots.engin.umich.edu/~aushani/tsf_data.tar.gz
      echo Trying secondary address
      if wget http://www.aushani.com/files/tsf_data.tar.gz
      then
        echo Got tarball
      else
        echo Couldn\'t connect to http://www.aushani.com/files/tsf_data.tar.gz
        echo Please contact aushani@gmail.com for assistance
        break
      fi
    fi

    tar xvpf tsf_data.tar.gz

    ;;
  *)
    # Get data snippet
    read -r -p "Get small snippet of data [y/N] " response
    case "$response" in
      [yY][eY][eS]|[yY])
        read -p "Enter a path for storing the data: " -i "$HOME" -e path

        mkdir -p $path
        cd $path

        echo Downloading data

        if wget http://robots.engin.umich.edu/~aushani/tsf_data_small.tar.gz
        then
          echo Got tarball
        else
          echo Couldn\'t connect to http://robots.engin.umich.edu/~aushani/tsf_data_small.tar.gz
          echo Trying secondary address
          if wget http://www.aushani.com/files/tsf_data_small.tar.gz
          then
            echo Got tarball
          else
            echo Couldn\'t connect to http://www.aushani.com/files/tsf_data_small.tar.gz
            echo Please contact aushani@gmail.com for assistance
            break
          fi
        fi

        tar xvpf tsf_data_small.tar.gz

        ;;
      *)
        echo Not getting data
    esac
    echo
    #######################################
esac
echo
#######################################

